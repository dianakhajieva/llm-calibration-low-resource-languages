"""
The main experiment loop (the diagram: question -> ask model -> record -> score).

Run from the project root AFTER load_data.py:
    python src/run_eval.py

For each model, each method and each question, this asks the model, saves the
raw reply, extracts the answer and the confidence, checks correctness, and writes
everything to results/processed/results.csv.

While testing, set sample_size to a small number (like 10) in config/config.yaml
so runs are fast and cheap. Raise it to ~500 for the real run.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, resolve_path
from prompts import build_answer_prompt, build_confidence_prompt
from parse import parse_answer, parse_confidence, letter_to_index
from models import get_model

import pandas as pd          # pip install pandas
from tqdm import tqdm        # pip install tqdm  (progress bars)

from dotenv import load_dotenv  # pip install python-dotenv
load_dotenv()  # reads API keys from a .env file in the project root


def read_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def save_raw(raw_dir, model_id, language_code, method, item_idx, text):
    """Save every raw reply, so you can re-parse later without paying again."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{model_id}__{language_code}__{method}__{item_idx}.txt"
    with open(raw_dir / fname, "w", encoding="utf-8") as f:
        f.write(text if text is not None else "")


def main():
    config = load_config()
    gen = config["generation"]
    methods = config["methods"]

    data_path = resolve_path(config["paths"]["data_sample"])
    if not data_path.exists():
        raise SystemExit(
            f"No data found at {data_path}. Run `python src/load_data.py` first."
        )
    records = read_records(data_path)
    raw_dir = resolve_path(config["paths"]["raw_outputs_dir"])
    results_path = resolve_path(config["paths"]["processed_results"])
    results_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_cfg in config["models"]:
        model_id = model_cfg["id"]
        supports_logprobs = model_cfg.get("supports_logprobs", False)
        print(f"\n=== Loading model: {model_id} ===")
        try:
            model = get_model(model_cfg)
        except Exception as e:
            print(f"Could not load {model_id}: {e}. Skipping this model.")
            continue

        for method in methods:
            if method == "logprob" and not supports_logprobs:
                print(f"Skipping 'logprob' for {model_id} (this model does not support it).")
                continue

            print(f"--- {model_id} | method = {method} ---")
            for rec in tqdm(records):
                passage, question, options = rec["passage"], rec["question"], rec["options"]
                predicted_letter, confidence, raw_text = None, None, ""
                try:
                    if method == "verbalized":
                        prompt = build_confidence_prompt(passage, question, options)
                        raw_text = model.generate(
                            prompt,
                            temperature=gen["temperature"],
                            max_new_tokens=gen["max_new_tokens"],
                        )
                        predicted_letter = parse_answer(raw_text)
                        confidence = parse_confidence(raw_text)
                    else:  # method == "logprob"
                        prompt = build_answer_prompt(passage, question, options)
                        predicted_letter, confidence = model.answer_with_token_logprob(prompt)
                        raw_text = f"{predicted_letter} (p={confidence})"
                except Exception as e:
                    raw_text = f"ERROR: {e}"

                save_raw(raw_dir, model_id, rec["language_code"], method, rec["item_idx"], raw_text)

                predicted_index = letter_to_index(predicted_letter) if predicted_letter else None
                is_correct = (
                    int(predicted_index == rec["correct_index"])
                    if predicted_index is not None
                    else None
                )

                rows.append({
                    "model_id": model_id,
                    "provider": model_cfg["provider"],
                    "method": method,
                    "language_code": rec["language_code"],
                    "language_name": rec["language_name"],
                    "item_idx": rec["item_idx"],
                    "item_id": rec["item_id"],
                    "predicted_letter": predicted_letter,
                    "predicted_index": predicted_index,
                    "correct_index": rec["correct_index"],
                    "is_correct": is_correct,
                    "confidence": confidence,
                    "parse_ok": int(predicted_letter is not None and confidence is not None),
                })

            # Save after every (model, method) so a crash never loses finished work.
            pd.DataFrame(rows).to_csv(results_path, index=False)
            print(f"Saved progress to {results_path}")

    print(f"\nDone. Final results are in {results_path}")


if __name__ == "__main__":
    main()
