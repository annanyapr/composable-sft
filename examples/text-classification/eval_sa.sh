#!/bin/bash

# Check if at least one argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <LANG> [TASK_FT] OR $0 <LANG_FT_PATH> <LANG> [TASK_FT]"
  exit 1
fi

if [ -d "$1" ]; then
  # Case: LANG_FT_PATH is provided
  LANG_FT=$1
  LANG=$2
  TASK_FT=${3:-"cambridgeltl/xlmr-task-sft-nusax_senti"}  # Optional, with a default value
else
  # Case: LANG is provided
  LANG=$1
  LANG_FT="cambridgeltl/xlmr-lang-sft-${LANG}-small"
  TASK_FT=${2:-"cambridgeltl/xlmr-task-sft-nusax_senti"}  # Optional, with a default value
fi

# Echo parameters
echo "LANG_FT: $LANG_FT"
echo "LANG: $LANG"
echo "TASK_FT: $TASK_FT"

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
