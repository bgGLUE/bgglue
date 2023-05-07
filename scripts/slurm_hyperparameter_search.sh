#!/bin/bash

set -xe

# bgGLUE project root in SLURM
HOME_PATH="PATH/TO/THE/PUBLIC/FOLDER"

# (Optional) WANDB API KEY
WANDB_API_KEY="YOUR_WANDB_API_KEY"

# (Optional) WANDB Project
WANDB_PROJECT="bglue"

# Path to the python ENV in SLURM
PYTHON_ENV_PATH="PATH/TO/PYTHON/bin/"

# Model names or paths, one per line
MODEL_NAMES=(
	microsoft/Multilingual-MiniLM-L12-H384
	distilbert-base-multilingual-cased
	bert-base-multilingual-cased
	xlm-roberta-base
	xlm-roberta-large
)

WANDB_WATCH=false

SEED=42
MAX_SEQ_LEN=512
BATCH_SIZE=16
EVAL_BATCH_SIZE=32
LEARNING_RATES=(
# 2--5 for all models
5e-05
4e-05
# XLM-Large start
3e-05
2e-05
# Only for XLM-Large (1--3)
1e-05
)
GRAD_ACC=1
WARMUP=0.06
WEIGHT_DEC=0.06

TASK_SCRIPTS=(
  bsnlp_trainer.job
  cinexio_regression_trainer.job
  clef_trainer.job
  crediblenews_trainer.job
  exams_trainer.job
  fakenews_trainer.job
  udep_trainer.job
  wikiann_trainer.job
  xnli_trainer.job
)
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
  for LEARNING_RATE in "${LEARNING_RATES[@]}"
  do
    for JOB in "${TASK_SCRIPTS[@]}"
    do
      sbatch -v --export=ALL,HOME_PATH=${HOME_PATH},MODEL_NAME=${MODEL_NAME},SEED=${SEED},MAX_SEQ_LEN=${MAX_SEQ_LEN},BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},LEARNING_RATE=${LEARNING_RATE},GRAD_ACC=${GRAD_ACC},WARMUP=${WARMUP},WEIGHT_DEC=${WEIGHT_DEC},WANDB_API_KEY=${WANDB_API_KEY},WANDB_PROJECT=${WANDB_PROJECT},WANDB_WATCH=${WANDB_WATCH},PYTHON_ENV_PATH=${PYTHON_ENV_PATH} \
          --partition gpu_ai --gres=gpu:1 -t 0-30:00 \
          scripts/${JOB}
    done
  done
done