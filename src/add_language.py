"""
Extend the fixed 500-question sample with an additional language,
without touching or re-sampling questions for which model responses
have already been collected in other languages.

This script:
1. Loads the existing sample file (e.g. data/sample/sample_500.csv).
2. Downloads the requested Belebele language config from Hugging Face.
3. Builds the same (link, question_number) composite key used by
   load_data.py to align languages.
4. Left-joins the new language onto the existing sample rows only,
   so the set of 500 questions never changes.
5. Saves the result, after first writing a timestamped backup of the
   original sample file.

Usage
-----
    python -m src.add_language --code kaz_Cyrl --prefix kk

`--prefix` is the short column prefix used for the new language's
columns (kk_passage, kk_question, kk_a1, ... kk_correct), matching the
existing en_/ru_/uz_ convention (English/Russian/Uzbek use their
ISO 639-1 codes, so Kazakh uses "kk").
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add a new aligned language to the fixed sample file."
    )
    parser.add_argument(
        "--code",
        type=str,
        required=True,
        help="Belebele language config name, e.g. 'kaz_Cyrl'.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Short column prefix for the new language, e.g. 'kk'.",
    )
    parser.add_argument(
        "--sample-file",
        type=str,
        default="data/sample/sample_500.csv",
        help="Path to the existing fixed sample CSV.",
    )
    return parser.parse_args()


def load_new_language(config_name):
    """Download one Belebele language config and return it as a DataFrame
    with a composite (link, question_number) key column."""

    ds = load_dataset("facebook/belebele", config_name)

    rows = []

    for row in ds["test"]:
        rows.append(
            {
                "key": f"{row['link']}__{row['question_number']}",
                "question": row["question"],
                "passage": row["flores_passage"],
                "a1": row["mc_answer1"],
                "a2": row["mc_answer2"],
                "a3": row["mc_answer3"],
                "a4": row["mc_answer4"],
                "correct": row["correct_answer_num"],
            }
        )

    return pd.DataFrame(rows)


def main():

    args = parse_args()

    sample_path = Path(args.sample_file)

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample file not found: {sample_path}. "
            f"Run src/load_data.py first."
        )

    sample = pd.read_csv(sample_path)

    if "en_key" not in sample.columns:
        raise ValueError(
            "Expected an 'en_key' column in the sample file (built as "
            "link__question_number) to align the new language onto. "
            "Check that this is the original sample_500.csv produced "
            "by load_data.py."
        )

    prefix = args.prefix

    new_col_names = [
        f"{prefix}_key",
        f"{prefix}_question",
        f"{prefix}_passage",
        f"{prefix}_a1",
        f"{prefix}_a2",
        f"{prefix}_a3",
        f"{prefix}_a4",
        f"{prefix}_correct",
    ]

    if any(col in sample.columns for col in new_col_names):
        raise ValueError(
            f"Columns with prefix '{prefix}_' already exist in the sample "
            f"file. Choose a different --prefix, or remove those columns "
            f"first before repeating this step."
        )

    print(f"Downloading Belebele config: {args.code}")
    new_lang = load_new_language(args.code)
    print(f"Downloaded {len(new_lang)} rows for {args.code}")

    new_lang = new_lang.add_prefix(f"{prefix}_")

    merged = sample.merge(
        new_lang,
        left_on="en_key",
        right_on=f"{prefix}_key",
        how="left",
    )

    missing = merged[f"{prefix}_question"].isna().sum()

    if missing > 0:
        print(
            f"WARNING: {missing} of {len(merged)} questions could not be "
            f"aligned to {args.code} (no matching link+question_number). "
            f"These rows will have empty {prefix}_* columns. Inspect "
            f"before running the evaluation pipeline on this language."
        )
    else:
        print(f"All {len(merged)} questions successfully aligned to {args.code}.")

    # Back up the original sample file before overwriting it.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = sample_path.with_name(
        f"{sample_path.stem}_backup_{timestamp}{sample_path.suffix}"
    )
    shutil.copy(sample_path, backup_path)
    print(f"Backed up original sample file to: {backup_path}")

    merged.to_csv(sample_path, index=False)
    print(f"Saved updated sample file to: {sample_path}")
    print(f"New columns added: {new_col_names}")


if __name__ == "__main__":
    main()