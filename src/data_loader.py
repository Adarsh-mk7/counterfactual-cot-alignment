"""
Data Loader & Formatting Utilities for Counterfactual CoT Alignment.
Handles ingestion, prompt formatting, validation, and Hugging Face Dataset creation for DPO.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple


def load_json_records(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSON or JSONL file into a list of dictionaries."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        # Check if JSON Lines
        first_line = f.readline().strip()
        f.seek(0)
        if first_line.startswith("{") and not first_line.endswith("}"):
            # Multi-line JSON
            return json.load(f)
        elif first_line.startswith("["):
            return json.load(f)
        else:
            # Attempt line-delimited JSON
            records = []
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
            return records


def format_cot_prompt(question: str, bias_hint: Optional[str] = None) -> str:
    """Formats a question into clean or biased CoT prompt."""
    if bias_hint:
        return f"Question: {question}\nNote: {bias_hint}\nAnswer: Let's think step by step."
    return f"Question: {question}\nAnswer: Let's think step by step."


def format_llama_chat(user_content: str, system_prompt: Optional[str] = None) -> str:
    """Formats prompt using Llama-3 instruction chat template."""
    prompt = "<|begin_of_text|>"
    if system_prompt:
        prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


def load_dpo_dataset(
    file_path: str,
    max_samples: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Loads DPO preference pairs (prompt, chosen, rejected) and returns
    a dictionary suitable for datasets.Dataset.from_dict.
    """
    records = load_json_records(file_path)
    if max_samples:
        records = records[:max_samples]

    prompts = []
    chosen = []
    rejected = []

    for item in records:
        if "prompt" in item and "chosen" in item and "rejected" in item:
            prompts.append(item["prompt"])
            chosen.append(item["chosen"])
            rejected.append(item["rejected"])
        else:
            raise KeyError(
                f"Missing required DPO keys in item. Expected 'prompt', 'chosen', 'rejected'. Found: {list(item.keys())}"
            )

    return {
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected
    }


def validate_evaluation_records(records: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Validates that evaluation records meet the required schema."""
    required_keys = {"clean_prompt", "biased_prompt", "ground_truth", "corrupt_hint"}
    for idx, r in enumerate(records):
        missing = required_keys - set(r.keys())
        if missing:
            return False, f"Record at index {idx} is missing keys: {missing}"
    return True, "All records valid."
