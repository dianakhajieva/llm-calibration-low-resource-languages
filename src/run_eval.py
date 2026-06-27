"""
Evaluation pipeline for multilingual calibration experiments.

Pipeline:

config.yaml
    ↓
sample_500.csv
    ↓
build_prompt()
    ↓
provider.generate()
    ↓
parse_response()
    ↓
prediction dictionary

This version uses a DummyProvider so the pipeline can be tested
without API keys.
"""

from pathlib import Path

import pandas as pd
import yaml

from prompts import build_prompt
from parse import parse_response
from models import get_provider


def load_config():
    """Load experiment configuration."""

    config_path = Path("config/config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(config):
    """Load benchmark dataset."""

    sample_path = config["paths"]["sample_file"]

    return pd.read_csv(sample_path)


def build_first_prompt(df):
    """Create a prompt from the first benchmark question."""

    row = df.iloc[0]

    prompt = build_prompt(
        passage=row["en_passage"],
        question=row["en_question"],
        a1=row["en_a1"],
        a2=row["en_a2"],
        a3=row["en_a3"],
        a4=row["en_a4"],
    )

    return prompt


def main():

    # ------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------

    config = load_config()

    print("=" * 60)
    print("Configuration loaded")
    print("=" * 60)

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------

    df = load_dataset(config)

    print(f"Dataset : {config['dataset']['name']}")
    print(f"Questions: {len(df)}")

    # ------------------------------------------------------------
    # Load provider
    # ------------------------------------------------------------

    model_cfg = config["models"][0]

    provider = get_provider(
        provider_name=model_cfg["provider"],
        model_name=model_cfg["model_name"],
    )

    print(f"Provider : {model_cfg['provider']}")
    print(f"Model    : {provider.model_name}")

    # ------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------

    prompt = build_first_prompt(df)

    print("\n" + "=" * 60)
    print("Prompt")
    print("=" * 60)
    print(prompt)

    # ------------------------------------------------------------
    # Generate response
    # ------------------------------------------------------------

    response = provider.generate(prompt)

    print("\n" + "=" * 60)
    print("Raw Model Response")
    print("=" * 60)
    print(response)

    # ------------------------------------------------------------
    # Parse response
    # ------------------------------------------------------------

    parsed = parse_response(response)

    print("\n" + "=" * 60)
    print("Parsed Response")
    print("=" * 60)
    print(parsed)


if __name__ == "__main__":
    main()