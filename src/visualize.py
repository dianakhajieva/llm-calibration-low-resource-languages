"""
Generate publication-quality visualizations for multilingual
LLM calibration experiments.

Figures generated
-----------------
1. Reliability diagrams
2. Confidence histograms
3. Accuracy comparison
4. Expected Calibration Error comparison

Author:
Multilingual LLM Calibration Project
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------


def load_config():
    """Load project configuration."""

    config_path = Path("config/config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------


def load_metrics(config):
    """
    Load metrics.csv produced by compute_metrics.py.
    """

    metrics_path = Path(
        config["paths"]["processed_results"]
    )

    return pd.read_csv(metrics_path)


def load_predictions(config):
    """
    Load all raw prediction CSV files.
    """

    raw_dir = Path(
        config["paths"]["raw_outputs_dir"]
    )

    files = sorted(
        raw_dir.glob("*_results.csv")
    )

    frames = []

    for file in files:

        frames.append(
            pd.read_csv(file)
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------


def get_figure_directory():
    """
    Create results/figures if it does not exist.
    """

    figure_dir = Path(
        "results/figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return figure_dir


# ---------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------


def reliability_diagram(
    df,
    model,
    language,
    output_dir,
    n_bins=15,
):
    """
    Plot reliability diagram for one
    model-language pair.
    """

    subset = df[
        (df["model"] == model)
        &
        (df["language"] == language)
    ].copy()

    confidence = (
        subset["confidence"] / 100
    ).to_numpy()

    correctness = (
        subset["is_correct"]
        .astype(int)
        .to_numpy()
    )

    bins = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    bin_centers = []
    accuracies = []

    for lower, upper in zip(
        bins[:-1],
        bins[1:],
    ):

        if upper == 1.0:

            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        if np.sum(mask) == 0:
            continue

        bin_centers.append(
            confidence[mask].mean()
        )

        accuracies.append(
            correctness[mask].mean()
        )

    plt.figure(figsize=(6, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        linewidth=2,
        label="Perfect calibration",
    )

    plt.plot(
        bin_centers,
        accuracies,
        marker="o",
        linewidth=2,
        label=model,
    )

    plt.xlabel("Confidence")

    plt.ylabel("Accuracy")

    plt.title(
        f"{model} - {language}"
    )

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    filename = (
        output_dir
        /
        f"{model}_{language}_reliability.png"
    )

    plt.savefig(
        filename,
        dpi=300,
    )

    plt.close()
# ---------------------------------------------------------
# Confidence histogram
# ---------------------------------------------------------


def confidence_histogram(
    df,
    model,
    language,
    output_dir,
):
    """
    Plot confidence distribution for one
    model-language pair.
    """

    subset = df[
        (df["model"] == model)
        &
        (df["language"] == language)
    ]

    confidence = subset["confidence"]

    plt.figure(figsize=(7, 5))

    plt.hist(
        confidence,
        bins=10,
        edgecolor="black",
    )

    plt.xlabel("Confidence (%)")

    plt.ylabel("Count")

    plt.title(
        f"{model} - {language}"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    filename = (
        output_dir
        /
        f"{model}_{language}_confidence_histogram.png"
    )

    plt.savefig(
        filename,
        dpi=300,
    )

    plt.close()


# ---------------------------------------------------------
# Accuracy comparison
# ---------------------------------------------------------


def accuracy_barplot(
    metrics,
    output_dir,
):
    """
    Compare model accuracy using
    Overall metrics only.
    """

    overall = metrics[
        metrics["language"] == "Overall"
    ].copy()

    overall = overall.sort_values(
        "accuracy",
        ascending=False,
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        overall["model"],
        overall["accuracy"],
    )

    plt.ylim(0, 1)

    plt.ylabel("Accuracy")

    plt.xlabel("Model")

    plt.title(
        "Overall Accuracy by Model"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "accuracy_comparison.png",
        dpi=300,
    )

    plt.close()


# ---------------------------------------------------------
# Expected Calibration Error comparison
# ---------------------------------------------------------


def ece_barplot(
    metrics,
    output_dir,
):
    """
    Compare Expected Calibration Error
    across models.
    """

    overall = metrics[
        metrics["language"] == "Overall"
    ].copy()

    overall = overall.sort_values(
        "ece",
        ascending=True,
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        overall["model"],
        overall["ece"],
    )

    plt.ylabel("ECE")

    plt.xlabel("Model")

    plt.title(
        "Expected Calibration Error by Model"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "ece_comparison.png",
        dpi=300,
    )

    plt.close()
# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():

    config = load_config()

    metrics = load_metrics(config)

    predictions = load_predictions(config)

    output_dir = get_figure_directory()

    print("=" * 70)
    print("Generating figures...")
    print("=" * 70)

    # -----------------------------------------------------
    # Reliability diagrams
    # -----------------------------------------------------

    combinations = (
        predictions[
            ["model", "language"]
        ]
        .drop_duplicates()
        .sort_values(
            ["model", "language"]
        )
    )

    for _, row in combinations.iterrows():

        reliability_diagram(
            predictions,
            row["model"],
            row["language"],
            output_dir,
        )

    print("✓ Reliability diagrams")

    # -----------------------------------------------------
    # Confidence histograms
    # -----------------------------------------------------

    for _, row in combinations.iterrows():

        confidence_histogram(
            predictions,
            row["model"],
            row["language"],
            output_dir,
        )

    print("✓ Confidence histograms")

    # -----------------------------------------------------
    # Overall comparisons
    # -----------------------------------------------------

    accuracy_barplot(
        metrics,
        output_dir,
    )

    print("✓ Accuracy comparison")

    ece_barplot(
        metrics,
        output_dir,
    )

    print("✓ ECE comparison")

    print()

    print("=" * 70)
    print("Finished!")
    print(f"Figures saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()