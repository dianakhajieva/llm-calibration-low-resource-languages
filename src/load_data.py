"""
Download the Belebele benchmark and save a fixed, reproducible sample of
parallel questions (the same questions in English, Russian and Uzbek).

Run from the project root:
    python src/load_data.py

This writes data/sample/belebele_sample.jsonl, where each line is one question
in one language. Questions that belong together across languages share the same
'item_id' and 'item_idx', so you can group them later.
"""
import json
import random
import sys
import os

# Make sibling modules (utils) importable no matter where you run this from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, resolve_path

from datasets import load_dataset  # pip install datasets

HF_DATASET = "facebook/belebele"


def make_key(example):
    """A stable identifier shared by the same question across languages."""
    return f"{example['link']}__{example['question_number']}"


def load_language(code):
    """Load one language split as a dict: key -> example."""
    ds = load_dataset(HF_DATASET, code, split="test")
    return {make_key(ex): ex for ex in ds}


def main():
    config = load_config()
    seed = config["seed"]
    sample_size = config["sample_size"]
    languages = config["languages"]

    print(f"Loading {len(languages)} languages from {HF_DATASET} ...")
    per_language = {lang["code"]: load_language(lang["code"]) for lang in languages}

    # Keep only the questions that exist in EVERY language, so every sampled
    # item is fully parallel.
    first_code = languages[0]["code"]
    common_keys = set(per_language[first_code].keys())
    for code, data in per_language.items():
        common_keys &= set(data.keys())
    common_keys = sorted(common_keys)
    print(f"{len(common_keys)} questions are available in all languages.")

    # Sample reproducibly: the same seed always picks the same questions.
    rng = random.Random(seed)
    n = min(sample_size, len(common_keys))
    sampled_keys = rng.sample(common_keys, n)
    print(f"Sampling {n} questions (seed={seed}).")

    out_path = resolve_path(config["paths"]["data_sample"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for item_idx, key in enumerate(sampled_keys):
            for lang in languages:
                ex = per_language[lang["code"]][key]
                options = [ex["mc_answer1"], ex["mc_answer2"], ex["mc_answer3"], ex["mc_answer4"]]
                record = {
                    "item_id": key,
                    "item_idx": item_idx,
                    "language_code": lang["code"],
                    "language_name": lang["name"],
                    "passage": ex["flores_passage"],
                    "question": ex["question"],
                    "options": options,
                    # Belebele stores the correct answer as a string "1".."4";
                    # we convert it to a 0-based index (0..3).
                    "correct_index": int(ex["correct_answer_num"]) - 1,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved sample to {out_path}")
    print("TIP: open that file and read a few Uzbek questions to confirm they look right.")


if __name__ == "__main__":
    main()
