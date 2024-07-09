# P-pruning
Official code for LREC-COLING2024 paper "Pruning before Fine-tuning: A Retraining-free Compression Framework for Pre-trained Language Models"[[paper link](https://aclanthology.org/2024.lrec-main.1162/)]
<div align="center">
  <img src=figures/overview.pdf>
</div>


# Finetuning Settings
The experimental settings for finetuning the pruned models are listed as below.
| Model |                        Task                        |   learning rate  | batch size |  epoch  |
|:-----:|:--------------------------------------------------:|:----------------:|:----------:|:-------:|
|  bert | `stsb`, `mrpc`, `sst2`, `qnli`, `qqp`, `mnli`, `squad`, `squad_v2` | 5e-5, 3e-5, 2e-5 |   16, 32   | 2, 3, 4 |
|  gpt2 |                           `penn_treebank`                          |       1e-4       |     32     |    10   |
|       |                           `wikitext-2-v1`                          |       5e-4       |     32     |    8    |
|       |                          `wikitext-103-v1`                         |       1e-4       |     32     |    10   |


# Stage 1: Prune models
The target model is first pruned with `./mask.py` to obtain the `head_mask.pt` and `neuron_mask.pt`. The masks are saved at `/PATH/TO/OUTPUT/$TASK`.

    cd /PATH/TO/P-pruning
    MODEL=bert
    TASK=stsb
    SAVE_DIR=/PATH/TO/OUTPUT/$TASK
    python ./mask.py \
        --model $MODEL \
        --save_dir $SAVE_DIR \
        --task $TASK \
        --max_instances 3000 \
        --pruning_ratio 0.3 \
        --sampling_tokens 200 \
        --lambda_val 5

  # Stage 2: Finetune models
  The pruned target model is then finetuned towards the downstream task with `./finetune.py`.

    python /GPFS/data/pingjiewang/P-pruning/finetune.py \
        --model_name_or_path $MODEL \
        --task_name $TASK_NAME \
        --mask_dir $MASK_DIR \
        --output_dir $OUTPUT_DIR \
        --seed 3407 \
        --per_device_train_batch_size 16 \
        --learning_rate 2e-5 \
        --num_train_epochs 4.0

# Citation
```bibtex
@inproceedings{wang2024pruning,
              title={Pruning before Fine-tuning: A Retraining-free Compression Framework for Pre-trained Language Models},
              author={Wang, Pingjie and Liu, Hongcheng and Wang, Yanfeng and Wang, Yu},
              booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
              pages={13279--13289},
              year={2024}
            }
```
