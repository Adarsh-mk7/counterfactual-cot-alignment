import os
import sys
import random

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.perturb import (
    extract_gsm8k_answer,
    synthesize_corrupt_answer,
    generate_dpo_pair,
    BIAS_TEMPLATES
)
from src.evaluate import parse_final_number
from src.data_loader import format_cot_prompt, validate_evaluation_records


class TestPerturbationSuite:
    def test_extract_gsm8k_answer_standard(self):
        text = "Natalia sold 48/2 = 24 clips in May.\n#### 72"
        assert extract_gsm8k_answer(text) == "72"

    def test_extract_gsm8k_answer_with_commas(self):
        text = "Total revenue is 150000 - 130000 = $20,000.\n#### 20,000"
        assert extract_gsm8k_answer(text) == "20000"

    def test_extract_gsm8k_answer_negative(self):
        text = "The net loss is #### -45"
        assert extract_gsm8k_answer(text) == "-45"

    def test_extract_gsm8k_answer_decimal(self):
        text = "Average value is #### 12.50"
        assert extract_gsm8k_answer(text) == "12.50"

    def test_extract_gsm8k_answer_missing(self):
        assert extract_gsm8k_answer("The answer is 42 without delimiter.") == ""
        assert extract_gsm8k_answer("") == ""
        assert extract_gsm8k_answer(None) == ""

    def test_synthesize_corrupt_answer(self):
        rng = random.Random(123)
        gt = "50"
        corrupt = synthesize_corrupt_answer(gt, rng=rng)
        assert corrupt != gt
        assert int(corrupt) > 0  # Should stay positive
        
        # Test float corrupted
        gt_float = "12.5"
        corrupt_float = synthesize_corrupt_answer(gt_float, rng=rng)
        assert corrupt_float != gt_float

    def test_bias_templates_formatting(self):
        assert len(BIAS_TEMPLATES) >= 3
        corrupt_ans = "42"
        for tpl in BIAS_TEMPLATES:
            formatted = tpl.format(corrupt_ans=corrupt_ans)
            assert corrupt_ans in formatted
            assert "{" not in formatted and "}" not in formatted

    def test_generate_dpo_pair(self):
        q = "A store has 15 apples and sells 6."
        gt = "9"
        corrupt = "11"
        hint = "I think the answer is 11."
        pair = generate_dpo_pair(q, gt, corrupt, hint)

        assert "prompt" in pair
        assert "chosen" in pair
        assert "rejected" in pair
        assert corrupt in pair["prompt"]
        assert gt in pair["chosen"]
        assert "incorrect" in pair["chosen"].lower()
        assert corrupt in pair["rejected"]


class TestEvaluationParsing:
    def test_parse_final_number_standard_phrases(self):
        text1 = "After careful thought, the final answer is: 130"
        assert parse_final_number(text1) == "130"

        text2 = "We calculated everything. Therefore, the answer is 42."
        assert parse_final_number(text2) == "42"

        text3 = "Here is the calculation.\n#### 624"
        assert parse_final_number(text3) == "624"

    def test_parse_final_number_with_dollar_sign(self):
        text = "She earned 50 * 0.2 = $10. So the final answer is $10"
        assert parse_final_number(text) == "10"

    def test_parse_final_number_trailing_numeric(self):
        text = "Step 1: 5 * 2 = 10\n10"
        assert parse_final_number(text) == "10"

    def test_parse_final_number_empty(self):
        assert parse_final_number("No numeric answer in this text.") == ""
        assert parse_final_number("") == ""


class TestDataLoader:
    def test_format_cot_prompt(self):
        clean = format_cot_prompt("What is 2+2?")
        assert "Question: What is 2+2?" in clean
        assert "Let's think step by step" in clean
        assert "Note:" not in clean

        biased = format_cot_prompt("What is 2+2?", bias_hint="I think it's 5.")
        assert "Note: I think it's 5." in biased

    def test_validate_evaluation_records(self):
        valid_records = [{
            "clean_prompt": "Q: ..",
            "biased_prompt": "Q: .. Note: ..",
            "ground_truth": "4",
            "corrupt_hint": "5"
        }]
        is_valid, _ = validate_evaluation_records(valid_records)
        assert is_valid is True

        invalid_records = [{"clean_prompt": "Q: .."}]
        is_valid, msg = validate_evaluation_records(invalid_records)
        assert is_valid is False
        assert "missing keys" in msg.lower()
