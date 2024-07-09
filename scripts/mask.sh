

cd /PATH/TO/P-pruning

# Supported models
# bert & gpt2
MODEL=bert

# Supported tasks
# bert: mrpc stsb qnli sst2 qqp mnli squad squad_v2
# gpt2: penn_treebank wikitext-2-v1 wikitext-103-v1
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
