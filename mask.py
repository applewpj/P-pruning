import os, sys
import torch
import argparse, random
import pdb
from transformers import AutoTokenizer, AutoModel, default_data_collator
from torch.utils.data import DataLoader, Subset
from src.utils.log import create_logger
from src.utils.extract import extract_config, extract_key
from src.utils.submodule import Architecture
from src.utils.flops import flops_per_head, flops_per_neuron
from src.utils.data import get_avg_seq_length, get_dataset, get_sequence_tasks, get_qa_tasks, get_lm_tasks
from src.mask.cluster import derive_head_tree, derive_neuron_tree, calculate_retained_head, calculate_retained_neuron
from src.mask.collect import collect_hidden_states


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/GPFS/data/pingjiewang/FLAD_BERT_BASE/model/bert-base-uncased", type=str)
parser.add_argument("--data_dir", default=None, type=str)
parser.add_argument("--save_dir", default="/GPFS/data/pingjiewang/P-pruning/output", type=str)
parser.add_argument("--task", default="stsb", type=str, choices=["stsb", "mrpc", "sst2", "qnli", "qqp", "mnli", "squad", "squad_v2", "penn_treebank", "wikitext-2-v1", "wikitext-103-v1"])
parser.add_argument("--max_instances", default=3000, type=int)
parser.add_argument("--pruning_ratio", default=0.5, type=float)
parser.add_argument("--sampling_tokens", default=200, type=int)
parser.add_argument("--lambda_val", default=5, type=float)
# parser.add_argument("--device", default="cpu", type=str)



if __name__ == "__main__":
    args = parser.parse_args()
    
    # Create logger
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logging_path = os.path.join(args.save_dir, f"mask.log")
    logger = create_logger(logging_path)
    logger.info(f"Masking parameters: {args}")
    
    # Prepare model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to("cuda")
    model_type = "bert" if "bert" in args.model else "gpt2"
    model_config = extract_config(model_type, model)
    model_key = extract_key(model_type)
    arch = Architecture(model, model_key, model_config)
    
    # Prepare dataset
    train_dataset = get_dataset(args.task, tokenizer, subset="train")
    if args.task in get_sequence_tasks():
        train_dataset = train_dataset.remove_columns('label')
    elif args.task in get_qa_tasks():
        train_dataset = train_dataset.remove_columns(['start_positions', 'end_positions'])
    elif args.task in get_lm_tasks():
        train_dataset = train_dataset.remove_columns('labels')
    if args.max_instances <= len(train_dataset):
        train_dataset = Subset(train_dataset, random.sample(range(len(train_dataset)), args.max_instances))  # Sample the dataset
    collate_fn = default_data_collator  # DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False, pin_memory=True)  
    
    # Collect hidden representations
    ATTN_O_input, FFN_2_input = collect_hidden_states(arch, dataloader)
    total_tokens, _ = ATTN_O_input[0].shape
    sampling_indices = random.sample(range(total_tokens), args.sampling_tokens)
    for layer_idx in range(model_config["layer_num"]):
        ATTN_O_input[layer_idx] = ATTN_O_input[layer_idx][sampling_indices]
        FFN_2_input[layer_idx] = FFN_2_input[layer_idx][sampling_indices]

    # Cluster the hidden representations and derive clustering trees
    attn_distance, attn_trees = derive_head_tree(model_config, ATTN_O_input)
    ffn_distance, ffn_trees = derive_neuron_tree(model_config, FFN_2_input)
    
    
    # Calculate total FLOPs
    seq_len = get_avg_seq_length(args.task)
    attn_FLOPs = flops_per_head(seq_len, model_config["d_emb"], model_config["head_size"]) * model_config["head_num"] * model_config["layer_num"]
    ffn_FLOPs = flops_per_neuron(seq_len, model_config["d_emb"]) * model_config["d_ffn"] * model_config["layer_num"]
    total_FLOPs = attn_FLOPs + ffn_FLOPs

    # Calculate constraint FLOPs
    pruned_attn_FLOPs = (1 - args.pruning_ratio) * total_FLOPs / (args.lambda_val + 1)
    pruned_ffn_FLOPs = args.lambda_val * pruned_attn_FLOPs
    target_attn_FLOPs = attn_FLOPs - pruned_attn_FLOPs
    target_ffn_FLOPs = ffn_FLOPs - pruned_ffn_FLOPs
    attn_FLOPs_ratio = target_attn_FLOPs / attn_FLOPs
    ffn_FLOPs_ratio = target_ffn_FLOPs / ffn_FLOPs

    # Calculate head mask and neuron mask
    th_attn, head_mask = calculate_retained_head(seq_len, model_config, target_attn_FLOPs, attn_trees, ATTN_O_input)
    th_ffn, neuron_mask = calculate_retained_neuron(seq_len, model_config, target_ffn_FLOPs, ffn_trees, FFN_2_input)
    torch.save(head_mask, os.path.join(args.save_dir, "head_mask.pt"))
    torch.save(neuron_mask, os.path.join(args.save_dir, "neuron_mask.pt"))
    
    
    remained_FLOPs = flops_per_head(seq_len, model_config["d_emb"], model_config["head_size"]) * torch.sum(head_mask) + flops_per_neuron(seq_len, model_config["d_emb"]) * torch.sum(neuron_mask)
    logger.info("Remained FLOPs: {:.2f}%".format(remained_FLOPs / total_FLOPs * 100))
    logger.info("Pruned Heads: {} ({:.2f}%)".format((model_config["head_num"] * model_config["layer_num"] - torch.sum(head_mask).item())*100, torch.sum(head_mask).item()/head_mask.numel()))
    logger.info("Pruned Neurons: {} ({:.2f}%)".format((model_config["d_ffn"] * model_config["layer_num"] - torch.sum(neuron_mask).item())*100, torch.sum(neuron_mask).item()/neuron_mask.numel()))
    logger.info(f"head mask and neuron mask have been saved at: {args.save_dir}")
    
    
