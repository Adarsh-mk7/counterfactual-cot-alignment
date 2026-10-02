#!/usr/bin/env bash
set -e

echo "============================================================"
echo "Counterfactual CoT Alignment - Full Benchmark Pipeline"
echo "============================================================"

# Step 1: Generate adversarial perturbations
echo "[1/2] Generating adversarial counterfactual perturbations..."
python src/perturb.py --num_samples 500 \
    --output_path "data/perturbed/gsm8k_sycophancy.json" \
    --dpo_output_path "data/perturbed/gsm8k_sycophancy_dpo_pairs.json"

# Step 2: Run inference & evaluation
echo "[2/2] Running inference evaluation and sycophancy metric engine..."
MODEL_ID=${1:-"meta-llama/Meta-Llama-3-8B-Instruct"}

python src/evaluate.py \
    --model_name "${MODEL_ID}" \
    --data_path "data/perturbed/gsm8k_sycophancy.json" \
    --output_metrics "results/eval_metrics.json" \
    --output_predictions "results/predictions.jsonl"

echo "============================================================"
echo "Benchmark completed successfully! Results in results/eval_metrics.json"
echo "============================================================"
