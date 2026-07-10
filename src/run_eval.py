"""
Evaluation pipeline for multilingual calibration experiments.

Current version:
- Loads configuration
- Loads Belebele sample
- Runs all configured models
- Evaluates multiple languages
- Parses model responses
- Saves one CSV per model
"""

from pathlib import Path

import pandas as pd
import yaml

from src.prompts import build_prompt
from src.parse import parse_response
from src.models import get_provider

ANSWER_MAP = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
}

def load_config():
    config_path = Path("config/config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(config):
    return pd.read_csv(config["paths"]["sample_file"])


def build_language_prompt(row, language):
    return build_prompt(
        passage=row[f"{language}_passage"],
        question=row[f"{language}_question"],
        a1=row[f"{language}_a1"],
        a2=row[f"{language}_a2"],
        a3=row[f"{language}_a3"],
        a4=row[f"{language}_a4"],
    )


def main():

    config = load_config()

    df = load_dataset(config)

    LANGUAGE_PREFIX = {
        "eng_Latn": "en",
        "rus_Cyrl": "ru",
        "uzn_Latn": "uz",
    }
    
    # Small test while developing
    test_size = min(2, config["sample_size"])

    for model_cfg in config["models"]:

        print("=" * 70)
        print(f"Model: {model_cfg['id']}")
        print("=" * 70)

        provider = get_provider(
            model_cfg["provider"],
            model_cfg["model_name"]
        )

        results = []

        for language in config["languages"]:

            lang_name = language["name"]
            lang_code = LANGUAGE_PREFIX[language["code"]]

            print("=" * 70)
            print(f"Running {lang_name}")
            print("=" * 70)

            for idx, row in df.head(test_size).iterrows():

                print(f"Question {idx + 1}")

                prompt = build_language_prompt(row, lang_code)

                raw_response = provider.generate(prompt)

                parsed = parse_response(raw_response)

                if parsed is None:
                    print("Could not parse response:")
                    print(raw_response)
                    continue

                predicted = parsed["answer"]
                confidence = parsed["confidence"]

                correct_number = str(row[f"{lang_code}_correct"])
                correct_letter = ANSWER_MAP[correct_number]

                results.append({
                    "question_id": idx + 1,
                    "language": lang_name,
                    "model": model_cfg["id"],
                    "predicted_answer": predicted,
                    "confidence": confidence,
                    "correct_number": correct_number,
                    "correct_answer": correct_letter,
                    "is_correct": predicted == correct_letter,
                })

                print(
                    f"Predicted={predicted} | "
                    f"Correct={correct_letter} | "
                    f"Confidence={confidence}"
                )

            print()

        results_df = pd.DataFrame(results)

        output_dir = Path(config["paths"]["raw_outputs_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{model_cfg['id']}_results.csv"

        results_df.to_csv(output_file, index=False)

        print("=" * 70)
        print(f"Finished {model_cfg['id']}")
        print(f"Saved results to: {output_file}")
        print("=" * 70)
        print()


if __name__ == "__main__":
    main()