#!/bin/bash
#
#SBATCH -N 1                      # number of nodes
#SBATCH -n 4                      # number of cores
#SBATCH -o logs/fakenews/slurm.%N.%j.out        # STDOUT
#SBATCH -e logs/fakenews/slurm.%N.%j.err        # STDER

set -eux

ACTION_STEPS=200
EPOCHS=20
TASK_NAME=fakenews-new

#MODEL_NAME=xlm-roberta-base
#SEED=42
#MAX_SEQ_LEN=384
#BATCH_SIZE=32
#EVAL_BATCH_SIZE=128
#LEARNING_RATE=5e-05
#GRAD_ACC=1
#WARMUP=0.06
#WEIGHT_DEC=0.06
OUTPUT_DIR="${HOME_PATH}models/${TASK_NAME}/${MODEL_NAME}/seed_${SEED}/ep_${EPOCHS}_bs_${BATCH_SIZE}_ga_${GRAD_ACC}_lr_${LEARNING_RATE}_seq_${MAX_SEQ_LEN}_warm_${WARMUP}_weight_${WEIGHT_DEC}"

export WANDB_TAGS="${MODEL_NAME},${TASK_NAME}"

if [[ -d "${OUTPUT_DIR}/pytorch_model.bin" ]]; then
  echo "Skipping directory model already exists"
  exit 0
fi

${PYTHON_ENV_PATH}python ${HOME_PATH}src/bg_glue_benchmark/run_classification.py \
--model_name_or_path ${MODEL_NAME} \
--dataset_name ${HOME_PATH}src/bg_glue_benchmark/bgdatasets/bgglue \
--dataset_config_name fakenews \
--data_dir ${HOME_PATH}data \
--first_text_column_name "title" \
--second_text_column_name "content" \
--metric_average "binary" \
--do_train \
--do_eval \
--do_predict \
--load_best_model_at_end \
--metric_for_best_model "f1" \
--warmup_ratio ${WARMUP} \
--save_strategy steps \
--save_steps ${ACTION_STEPS} \
--logging_strategy steps \
--logging_steps ${ACTION_STEPS} \
--evaluation_strategy steps \
--eval_steps ${ACTION_STEPS} \
--learning_rate ${LEARNING_RATE} \
--num_train_epochs ${EPOCHS} \
--max_seq_length ${MAX_SEQ_LEN} \
--output_dir ${OUTPUT_DIR} \
--weight_decay ${WEIGHT_DEC} \
--per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
--per_device_train_batch_size ${BATCH_SIZE} \
--gradient_accumulation_steps ${GRAD_ACC} \
--seed ${SEED} \
--fp16 \
--overwrite_output \
--overwrite_cache \
--cache_dir ${HOME_PATH}cache \
--save_total_limit 1 \
--report_to wandb \
--pad_to_max_length false
