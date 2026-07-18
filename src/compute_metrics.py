"""
Compute calibration metrics for multilingual LLM evaluation.

This script reads all prediction CSV files produced by run_eval.py,
computes calibration metrics for every model-language pair,
and saves the results as a single metrics table.

Metrics:
---------
- Accuracy
- Average Confidence
- Brier Score
- Expected Calibration Error (ECE)

Author:
Multilingual LLM Calibration Project
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def load_config():
    """
    Load experiment configuration.
    """

    config_path = Path("config/config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------


def load_prediction_files(config):
    """
    Load every prediction CSV from the raw_outputs directory.

    Returns
    -------
    pandas.DataFrame
        Combined dataframe containing all models.
    """

    raw_dir = Path(config["paths"]["raw_outputs_dir"])

    csv_files = sorted(raw_dir.glob("*_results.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No prediction files found in {raw_dir}"
        )

    frames = []

    for file in csv_files:

        df = pd.read_csv(file)

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------
# Metric Functions
# ---------------------------------------------------------------------


def accuracy(df):
    """
    Classification accuracy.

    Returns
    -------
    float
        Accuracy between 0 and 1.
    """

    return df["is_correct"].mean()


def average_confidence(df):
    """
    Mean verbalized confidence.

    Confidence values are converted from
    percentages to probabilities.

    Returns
    -------
    float
    """

    return (df["confidence"] / 100).mean()


def brier_score(df):
    """
    Compute Brier Score.

    Formula
    -------
    BS = mean((confidence - correctness)^2)

    Lower is better.
    Perfect calibration = 0.
    """

    confidence = df["confidence"] / 100

    correctness = (
        df["is_correct"]
        .astype(int)
        .astype(float)
    )

    return np.mean(
        (confidence - correctness) ** 2
    )


# ---------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------


def expected_calibration_error(
    df,
    n_bins=15,
):
    """
    Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    df : pandas.DataFrame

    n_bins : int
        Number of confidence bins.

    Returns
    -------
    float
    """

    confidence = (
        df["confidence"] / 100
    ).to_numpy()

    correctness = (
        df["is_correct"]
        .astype(int)
        .to_numpy()
    )

    bin_edges = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    ece = 0.0
    for lower, upper in zip(
        bin_edges[:-1],
        bin_edges[1:],
    ):

        if upper == 1.0:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        if np.sum(mask) == 0:
            continue

        bin_confidence = confidence[mask].mean()

        bin_accuracy = correctness[mask].mean()

        weight = np.sum(mask) / len(confidence)

        ece += weight * abs(
            bin_accuracy - bin_confidence
        )

    return ece


# ---------------------------------------------------------------------
# Compute Metrics
# ---------------------------------------------------------------------


def compute_metrics(df, config):
    """
    Compute metrics for every model-language pair
    and an overall summary for each model.
    """

    rows = []

    # Number of expected questions per language
    expected_per_language = config["sample_size"]

    # Total expected questions across all languages
    expected_overall = (
        config["sample_size"]
        * len(config["languages"])
    )

    # ---------------------------------------------------------
    # Per-language metrics
    # ---------------------------------------------------------

    grouped = df.groupby(
        ["model", "language"],
        sort=True,
    )

    for (model, language), group in grouped:

        answered = len(group)

        if answered == 0:
            continue

        rows.append(
            {
                "model": model,
                "language": language,
                "answered_questions": answered,
                "expected_questions": expected_per_language,
                "skipped_questions": expected_per_language - answered,
                "accuracy": accuracy(group),
                "average_confidence": average_confidence(group),
                "brier_score": brier_score(group),
                "ece": expected_calibration_error(group),
            }
        )

    # ---------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------

    for model, group in df.groupby(
        "model",
        sort=True,
    ):

        answered = len(group)

        if answered == 0:
            continue

        rows.append(
            {
                "model": model,
                "language": "Overall",
                "answered_questions": answered,
                "expected_questions": expected_overall,
                "skipped_questions": expected_overall - answered,
                "accuracy": accuracy(group),
                "average_confidence": average_confidence(group),
                "brier_score": brier_score(group),
                "ece": expected_calibration_error(group),
            }
        )

    metrics = pd.DataFrame(rows)

    metrics = metrics.sort_values(
        ["model", "language"]
    )

    metrics = metrics.round(
        {
            "accuracy": 4,
            "average_confidence": 4,
            "brier_score": 4,
            "ece": 4,
        }
    )

    return metrics.reset_index(drop=True)


# ---------------------------------------------------------------------
# Save Metrics
# ---------------------------------------------------------------------


def save_metrics(metrics, config):
    """
    Save metrics dataframe.
    """

    output_path = Path(
        config["paths"]["processed_results"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = metrics.round(
        {
            "accuracy": 4,
            "average_confidence": 4,
            "brier_score": 4,
            "ece": 4,
        }
    )

    metrics.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():

    config = load_config()

    predictions = load_prediction_files(config)

    metrics = compute_metrics(
      predictions,
      config,
   )

    output_path = save_metrics(
        metrics,
        config,
    )

    print("=" * 70)
    print("Calibration Metrics")
    print("=" * 70)
    print()

    print(metrics)

    print()

    print("=" * 70)
    print(f"Saved metrics to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()