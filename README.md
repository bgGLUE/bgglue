![alt text](logo.png "Title")
# bgGLUE: Bulgarian Language Evaluation Benchmark

bgGLUE (Bulgarian General Language Understanding Evaluation) is a benchmark for evaluating language models on Natural Language Understanding (NLU) tasks in Bulgarian. 
The benchmark includes NLU tasks targeting a variety of NLP problems (e.g., natural language inference, fact-checking, named entity recognition, sentiment analysis, question answering, etc.) and machine learning tasks (sequence labeling, document-level classification, and regression).

Leaderboard: https://bgglue.github.io/  
Models: https://huggingface.co/bgglue


## Getting Started

### 1. Install

#### Python

The `bgGLUE` project dependencies are managed with `poetry`.

To install poetry you need to run (official [instructions](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions)):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

*It is also possible to install it using `pip install poetry` but this is not recommended.*

The project requires `Python 3.8`. If this is the default in your environment you can use `poetry shell` to create a new venv. 

(Optional) you can use `codna` to create a new environment and set it as poetry's `env` for that project:

```
conda create -p envs/bg-glue-benchmark python=3.8

# This command needs to be run from a folder inside the bgglue project
poetry env use envs/bg-glue-benchmark
```


#### Installing the dependencies

Then run `poetry install`, if you want to include `weights & biases` you need to pass either `-E wandb`
or `--all-extras`.

If you run with SLURM you need to pass the full python env path to the job.
You can resolve it with`which python` inside your virtual environment. 
1. With poetry: `poetry run which python` or `poetry debug info`.
2. With conda you must activate the env with `conda activate /path/to/env`

## 2. Getting the Data

All the datasets are downloaded automatically by the dataset loader class. 
The archives with the bgGLUE datasets are provided under the [data/](data/) folder. 
Each dataset is provided as a `tar.gz` archive that contains a folder with the same name as the datasets (e.g., `bsnlp`), and three json lines (`jsonl`) files: `train.jsonl`, `dev.jsonl`, and `test.jsonl`.

If you need to use the CT21 offline, you can provide a path to a folder for the `CT21.T1` containing two archives 
`v1.zip` and `subtask-1a--bulgarian.zip`, downloaded from the [competition's repository](https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/tree/master/task1) 
under the folders `data` and `test-input`.

The `credible news` dataset is not public, and is released under a non-commercial licence ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)) 
for **research purposes only**. 
Participants need to agree with the "Terms of Use" described in this [Google form](https://forms.gle/Nm4T369eg8rXG3tx8), after that they will receive a download link.
You need to provide path to the directory where `credible_news.tar.gz` is store by passing the `--data_dir YOUR_DATA_PATH` argument.


## 3. Running Experiments 

### Running the Trainer Locally

We provide scripts for each task type. Note that one or more datasets can be run with the same datasets.
If you use `poetry`, all scripts should be run from inside a `poetry shell` or with `poetry run`.

The paths in the scripts below are resolved from the project root.

Example hyper-parameters:

```bash
export MODEL_NAME=xlm-roberta-base
export SEED=42
export MAX_SEQ_LEN=512
export BATCH_SIZE=32
export EVAL_BATCH_SIZE=128
export LEARNING_RATE=5e-05
export GRAD_ACC=1
export WARMUP=0.06
export WEIGHT_DEC=0.06
```

#### BSNLP

```bash
python src/bg_glue_benchmark/run_tag.py \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name src/bg_glue_benchmark/bgdatasets/bgglue \
    --dataset_config_name bsnlp \
    --do_train \
    --do_eval \
    --do_predict \
    --load_best_model_at_end \
    --metric_for_best_model "f1" \
    --task_name "ner" \
    --warmup_ratio ${WARMUP} \
    --save_strategy steps \
    --save_steps 200 \
    --logging_strategy steps \
    --logging_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --output_dir models/${MODEL_NAME} \
    --weight_decay ${WEIGHT_DEC} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --seed ${SEED} \
    --fp16 \
    --overwrite_output \
    --overwrite_cache \
    --cache_dir cache \
    --save_total_limit 1 \
    --report_to wandb \
    --pad_to_max_length false
```

#### Cinexio

```bash
python src/bg_glue_benchmark/run_cinexio.py \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name src/bg_glue_benchmark/bgdatasets/bgglue \
    --dataset_config_name cinexio \
    --do_train \
    --do_eval \
    --do_predict \
    --task_type regression \
    --load_best_model_at_end \
    --metric_for_best_model "sp_correlation" \
    --warmup_ratio ${WARMUP} \
    --save_strategy steps \
    --save_steps 200 \
    --logging_strategy steps \
    --logging_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --output_dir models/${MODEL_NAME} \
    --weight_decay ${WEIGHT_DEC} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --seed ${SEED} \
    --fp16 \
    --overwrite_output \
    --overwrite_cache \
    --cache_dir cache \
    --save_total_limit 1 \
    --report_to wandb \
    --pad_to_max_length false
```

#### CheckThat!'21 Task 1 Check-Worthiness (CT21.T1)

```bash
python src/bg_glue_benchmark/run_clef.py \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name src/bg_glue_benchmark/bgdatasets/bgglue \
    --dataset_config_name ct21t1 \
    --do_train \
    --do_eval \
    --do_predict \
    --load_best_model_at_end \
    --metric_for_best_model "avg_precision" \
    --warmup_ratio ${WARMUP} \
    --save_strategy steps \
    --save_steps 200 \
    --logging_strategy steps \
    --logging_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --output_dir models/${MODEL_NAME} \
    --weight_decay ${WEIGHT_DEC} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --seed ${SEED} \
    --fp16 \
    --overwrite_output \
    --overwrite_cache \
    --cache_dir cache \
    --save_total_limit 1 \
    --report_to wandb \
    --pad_to_max_length false
```

#### Credible News

```bash
python src/bg_glue_benchmark/run_classification.py \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name src/bg_glue_benchmark/bgdatasets/bgglue \
    --dataset_config_name crediblenews \
    --data_dir data \
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
    --save_steps 200 \
    --logging_strategy steps \
    --logging_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --output_dir models/${MODEL_NAME} \
    --weight_decay ${WEIGHT_DEC} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --seed ${SEED} \
    --fp16 \
    --overwrite_output \
    --overwrite_cache \
    --cache_dir cache \
    --save_total_limit 1 \
    --report_to wandb \
    --pad_to_max_length false
```


### EXAMS

```bash
python src/bg_glue_benchmark/run_multiple_choice.py \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name src/bg_glue_benchmark/bgdatasets/bgglue \
    --dataset_config_name examsbg \
    --do_train \
    --do_eval \
    --do_predict \
    --load_best_model_at_end \
    --metric_for_best_model "acc" \
    --warmup_ratio ${WARMUP} \
    --save_strategy steps \
    --save_steps 200 \
    --logging_strategy steps \
    --logging_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --output_dir models/${MODEL_NAME} \
    --weight_decay ${WEIGHT_DEC} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --seed ${SEED} \
    --fp16 \
    --overwrite_output \
    --overwrite_cache \
    --cache_dir cache \
    --save_total_limit 1 \
    --report_to wandb
```


#### Fake News


```bash
${PYTHON_ENV_PATH}python src/bg_glue_benchmark/run_classification.py \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name src/bg_glue_benchmark/bgdatasets/bgglue \
    --dataset_config_name fakenews \
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
    --save_steps 200 \
    --logging_strategy steps \
    --logging_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --output_dir models/${MODEL_NAME} \
    --weight_decay ${WEIGHT_DEC} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --seed ${SEED} \
    --fp16 \
    --overwrite_output \
    --overwrite_cache \
    --cache_dir cache \
    --save_total_limit 1 \
    --report_to wandb \
    --pad_to_max_length false
```

#### Universal Dependencies

```bash
python src/bg_glue_benchmark/run_tag.py \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name src/bg_glue_benchmark/bgdatasets/bgglue \
    --do_train \
    --do_eval \
    --do_predict \
    --load_best_model_at_end \
    --task_name "pos" \
    --metric_for_best_model "f1" \
    --warmup_ratio ${WARMUP} \
    --save_strategy steps \
    --save_steps 200 \
    --logging_strategy steps \
    --logging_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --output_dir models/${MODEL_NAME} \
    --weight_decay ${WEIGHT_DEC} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --seed ${SEED} \
    --fp16 \
    --overwrite_output \
    --overwrite_cache \
    --cache_dir cache \
    --save_total_limit 1 \
    --report_to wandb
```

#### PAN-X 

```bash
python src/bg_glue_benchmark/run_tag.py \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name src/bg_glue_benchmark/bgdatasets/bgglue \
    --dataset_config_name wikiannbg \
    --do_train \
    --do_eval \
    --do_predict \
    --load_best_model_at_end \
    --task_name "ner" \
    --metric_for_best_model "f1" \
    --warmup_ratio ${WARMUP} \
    --save_strategy steps \
    --save_steps 200 \
    --logging_strategy steps \
    --logging_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --output_dir models/${MODEL_NAME} \
    --weight_decay ${WEIGHT_DEC} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --seed ${SEED} \
    --fp16 \
    --overwrite_output \
    --overwrite_cache \
    --cache_dir cache \
    --save_total_limit 1 \
    --report_to wandb \
    --pad_to_max_length false
```


### XNLI

```bash
python src/bg_glue_benchmark/run_xnli.py \
    --dataset_name src/bg_glue_benchmark/bgdatasets/bgglue \
    --dataset_config_name xnlibg \
    --model_name_or_path ${MODEL_NAME} \
    --do_train \
    --do_eval \
    --do_predict \
    --load_best_model_at_end \
    --metric_for_best_model "accuracy" \
    --warmup_ratio ${WARMUP} \
    --save_strategy steps \
    --save_steps 200 \
    --logging_strategy steps \
    --logging_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --output_dir models/${MODEL_NAME} \
    --weight_decay ${WEIGHT_DEC} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --seed ${SEED} \
    --fp16 \
    --overwrite_output \
    --overwrite_cache \
    --cache_dir cache \
    --save_total_limit 1 \
    --report_to wandb \
    --pad_to_max_length false
```

### Running on SLURM
All scripts are in the `scripts/` folder and have a `.job` suffix. 
They are designed to work with SLURM and expect several ENV variables in order to work properly (see the next section).

The expected way to run them is using the `sbatch` command with `--export` option to fill the variables.
Finally, all jobs have a check if a model exists in the expected output folder, if yes then the process 
is aborted and the training script will NOT be executed.


#### Run Hyper-Parameter Tuning

We provide an automatic script that schedules sequentially experiments by looping over a set of pre-defined model names/paths and learning rates. 

Before running the script you need to edit the following variables in `scripts/slurm_hyperparameter_search.sh`:

1. `HOME_PATH="PATH/TO/THE/PUBLIC/FOLDER"` (the absolute path to the `bgglue` root folder)
2. `PYTHON_ENV_PATH="PATH/TO/PYTHON/bin/"` (path to the `bin` folder of your virtual env, expected to contain `python`)
3. `WANDB_API_KEY="YOUR_WANDB_API_KEY"` (your `weights & biases` API key)
4. `sbatch ... --partition PARTITION --gres=gpu:GPU_COUNT -t INTERVAL "PATH/TO/SCRIPTS"/${JOB}` -- SLURM parameters + the absolute path to the scripts folder.


```bash
bash scripts/slurm_hyperparameter_search.sh
```

## 4. Submitting to the Leaderboard

### Information Needed
We ask for seven pieces of information: A short name for your system, which will be displayed in the leaderboard. A URL for a paper or (if one is not available) website or code repository describing your system. A sentence or two describing your system. Make sure to mention any outside data or resources you use. A sentence or two explaining how you share parameters across tasks (or stating that you don't share parameters). The total number of trained parameters in your model. Do not count word or word-part embedding parameters, even if they are trained. The total number of trained parameters in your model that are shared across multiple tasks. If some parameters are shared across some but not all of your tasks, count those. Do not count word or word-part embedding parameters, even if they are trained. Whether you want your submission to be visible on the public leaderboard.

### Submission Format

Participants should submit an archive with `predictions.jsonl` files for each datasets:

```bash
# my_submission.zip

└── predictions
    ├── bsnlp
    │   └── predictions.jsonl
    ├── cinexio
    │   └── predictions.jsonl
    ├── crediblenews
    │   └── predictions.jsonl
    ├── ct21t1
    │   └── predictions.jsonl
    ├── examsbg
    │   └── predictions.jsonl
    ├── fakenews
    │   └── predictions.jsonl
    ├── udep
    │   └── predictions.jsonl
    ├── wikiannbg
    │   └── predictions.jsonl
    └── xnlibg
        └── predictions.jsonl
```

Each prediction row should contain two fields `index` -- the index of the example from the dataset, and `label` -- the predicted label (must be a valid label name or value).

CT21.T1 requires special fields:
```json
{"topic": "covid-19", "id": "1264909985177362438", "label": 1.0}
```

Cinexio requires the label to be a float value:
```json
{"index": 0, "label": 5.0}
```

NER/POS tagging datasets require a string with whitespace separated labels for each word.
```json
{"index": 0, "label": "O O O O O O O"}
```

### More Information

More information can be found in the ["FAQ" section](https://bgglue.github.io/faq/) on the bgGLUE website.

## References

Please cite as [[1]](https://arxiv.org/abs/2306.02349).

[1] Hardalov, M., Atanasova, P., Mihaylov, T., Angelova, G., Simov, K., Osenova, P., Stoyanov, V., Koychev, I., Nakov, P. and Radev, D., 2023. "*bgGLUE: A Bulgarian General Language Understanding Evaluation Benchmark*". Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

```
@inproceedings{hardalov-etal-2023-bgglue,
    title = "{bgGLUE}: A Bulgarian General Language Understanding Evaluation Benchmark",
    author = "Hardalov, Momchil and 
        Atanasova, Pepa and 
        Mihaylov, Todor and 
        Angelova, Galia and 
        Simov, Kiril and 
        Osenova, Petya and 
        Stoyanov, Ves and 
        Koychev, Ivan and 
        Nakov, Preslav and 
        Radev, Dragomir",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = july,
    year = "2023",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    address = "Toronto, Canada",
    url = "https://arxiv.org/abs/2306.02349"
}
```

Citations for the datasets can be found ["Task Information" page](https://bgglue.github.io/tasks/) on the bgGLUE website.

## License

This package is released under the [MIT License](LICENSE).

The primary bgGLUE tasks are built on and derived from existing datasets. We refer participants to the original licenses accompanying each dataset. For each dataset the license is listed on its ["Task Information" page](https://bgglue.github.io/tasks/) on the bgGLUE website.
