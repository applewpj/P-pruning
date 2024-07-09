cd /PATH/TO/P-pruning
MASK_DIR=/PATH/TO/OUTPUT/$TASK_NAME
OUTPUT_DIR=$MASK_DIR

# Supported models
# bert & gpt2
MODEL=bert

# Supported tasks
# bert: mrpc stsb qnli sst2 qqp mnli squad squad_v2
# gpt2: penn_treebank wikitext-2-v1 wikitext-103-v1
TASK=stsb

python ./finetune.py \
    --model_name_or_path $MODEL \
    --task_name $TASK_NAME \
    --mask_dir $MASK_DIR \
    --output_dir $OUTPUT_DIR \
    --seed 3407 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 4.0

