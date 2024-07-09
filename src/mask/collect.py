import logging
import torch
from src.utils.submodule import Architecture
from tqdm import tqdm

logger = logging.getLogger()



def hook_input(module, list_to_append, device="cpu"):
    hook = lambda _, inputs: list_to_append.append(inputs[0].to(device))
    handle = module.register_forward_pre_hook(hook)
    return handle



def collect_hidden_states(arch: Architecture, dataloader):
    model_config = arch.model_config
    
    device = arch.model.device
    arch.model.eval()
    
    handles = []
    ATTN_O_input, FFN_2_input = [], []
    for layer_idx in range(model_config["layer_num"]):
        ATTN_O_input.append([])
        FFN_2_input.append([])

    for layer_idx in range(model_config["layer_num"]):
        ATTN_O = arch.get_attn_o(layer_idx)
        FFN_2 = arch.get_ffn2(layer_idx)
        for module, cache in zip((ATTN_O, FFN_2), (ATTN_O_input[layer_idx], FFN_2_input[layer_idx])):
            handle = hook_input(module, cache)
            handles.append(handle)

    
    logger.info("Start collecting hidden states")
    attention_mask = []
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        attention_mask.append(batch['attention_mask'])
        for k, v in batch.items():
            batch[k] = v.to(device)
        with torch.no_grad():
            arch.model(**batch)
        

    attention_mask = torch.cat(attention_mask)
    for layer_idx in range(model_config["layer_num"]):
        ATTN_O_input[layer_idx] = torch.cat(ATTN_O_input[layer_idx])
        ATTN_O_input[layer_idx] = ATTN_O_input[layer_idx][attention_mask==1]
        FFN_2_input[layer_idx] = torch.cat(FFN_2_input[layer_idx])
        FFN_2_input[layer_idx] = FFN_2_input[layer_idx][attention_mask==1]
    
    for handle in handles:
        handle.remove()

    return ATTN_O_input, FFN_2_input