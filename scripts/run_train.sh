#!/usr/bin/env bash
set -e

CONFIG_PATH=${1:-"configs/dpo_config.yaml"}
DATA_PATH=${2:-"data/perturbed/gsm8k_sycophancy_dpo_pairs.json"}
OUTPUT_DIR=${3:-"./aligned_model_lora"}

echo "Starting Counterfactual DPO Training..."
echo "Configuration:       $CONFIG_PATH"
echo "Preference Data:     $DATA_PATH"
echo "Output Adapter:      $OUTPUT_DIR"

python src/train_dpo.py \
    --config "$CONFIG_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR"
