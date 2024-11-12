#!/bin/bash

# Check if language argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <LANG> [TASK_FT]"
  exit 1
fi

LANG=$1  # Required argument
LANG_FT="cambridgeltl/xlmr-lang-sft-${LANG}-small"
TASK_FT=${2:-"cambridgeltl/xlmr-task-sft-nusax_senti"}  # Optional, with a default value

RESULTS_DIR="results/NusaX-senti/${LANG}"
mkdir -p $RESULTS_DIR

# Run the Python script with specified parameters
python run_text_classification.py \
  --model_name_or_path xlm-roberta-base \
  --lang_ft $LANG_FT \
  --task_ft $TASK_FT \
  --dataset_name indonlp/NusaX-senti \
  --dataset_config_name $LANG \
  --output_dir $RESULTS_DIR \
  --do_eval \
  --eval_metric f1 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --eval_split test \
  --overwrite_output_dir
