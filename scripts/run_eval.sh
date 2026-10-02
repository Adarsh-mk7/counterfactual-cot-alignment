#!/usr/bin/env bash
set -e

# Default parameters
MODEL_NAME=${1:-"meta-llama/Meta-Llama-3-8B-Instruct"}
DATA_PATH=${2:-"data/perturbed/gsm8k_sycophancy.json"}
OUTPUT_METRICS=${3:-"results/eval_metrics.json"}
OUTPUT_PREDICTIONS=${4:-"results/predictions.jsonl"}

echo "Running Counterfactual CoT Evaluation..."
echo "Model:               $MODEL_NAME"
echo "Dataset:             $DATA_PATH"
echo "Output Metrics:      $OUTPUT_METRICS"

python src/evaluate.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_metrics "$OUTPUT_METRICS" \
    --output_predictions "$OUTPUT_PREDICTIONS"
