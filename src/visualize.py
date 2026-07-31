"""
Generate publication-quality visualizations for multilingual
LLM calibration experiments.

Figures generated
------------------
  1. reliability_diagrams_grid.(png|pdf)
        One figure, subplot grid = languages (rows) x models (cols).
        Each panel: reliability curve, shaded over/under-confidence gap,
        confidence-histogram inset, ECE + accuracy + n annotated.

  2. confidence_histograms_grid.(png|pdf)
        Same grid layout, confidence distribution per pair, with mean
        confidence and accuracy marked.

  3. accuracy_comparison.(png|pdf)
        Overall accuracy per model, sorted, with 95% bootstrap CI error
        bars (from compute_metrics.py) and value labels on bars.

  4. ece_comparison.(png|pdf)
        Overall ECE per model, sorted ascending (lower = better), with
        95% bootstrap CI error bars and value labels.

  5. ece_heatmap.(png|pdf)
        Model x language ECE heatmap, showing calibration error for
        every model-language combination at a glance.

If GENERATE_INDIVIDUAL = True, also writes one reliability diagram and
one confidence histogram per model-language pair, in addition to the
grids above.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

# Single source of truth for ECE: the same function the metrics use.
try:
    from src.compute_metrics import _ece_from_arrays
except ImportError:
    from compute_metrics import _ece_from_arrays

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colorblind-safe qualitative palette (Wong, 2011), reused everywhere
# so each model keeps the same color across every figure.
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]

MIN_BIN_SIZE = 10  # minimum samples for a reliability bin to be plotted
N_BINS = 15

GENERATE_INDIVIDUAL = False  # set True for supplementary per-pair figures


def model_color_map(models):
    """Stable color assignment: same model -> same color everywhere."""
    models_sorted = sorted(models)
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models_sorted)}


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
    """Load metrics.csv produced by compute_metrics.py."""

    metrics_path = Path(config["paths"]["processed_results"])
    return pd.read_csv(metrics_path)


def load_predictions(config):
    """Load all raw prediction CSV files."""

    raw_dir = Path(config["paths"]["raw_outputs_dir"])
    files = sorted(raw_dir.glob("*_results.csv"))
    frames = [pd.read_csv(file) for file in files]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------


def get_figure_directory(config=None):
    """Create the figures directory (from config if available) if needed."""

    if config is not None and "figures_dir" in config.get("paths", {}):
        figure_dir = Path(config["paths"]["figures_dir"])
    else:
        figure_dir = Path("results/figures")

    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def savefig_both(fig, output_dir, stem):
    """Save a figure as both PDF (vector, for LaTeX) and PNG (preview/slides)."""

    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=300)


# ---------------------------------------------------------
# Calibration bin computation (shared by several figures)
# ---------------------------------------------------------


def compute_bins(confidence, correctness, n_bins=N_BINS, min_bin_size=MIN_BIN_SIZE):
    """
    Bin (confidence, correctness) pairs into n_bins equal-width bins.

    Returns arrays of: bin_centers (mean confidence in bin), accuracies
    (mean correctness in bin), counts (n samples in bin), and the
    dataset-level ECE computed as the count-weighted average |acc-conf|
    gap over all bins meeting min_bin_size.
    """

    bins = np.linspace(0, 1, n_bins + 1)

    bin_centers, accuracies, counts = [], [], []

    for lower, upper in zip(bins[:-1], bins[1:]):

        if upper == 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = np.sum(mask)
        if count < min_bin_size:
            continue

        bin_centers.append(confidence[mask].mean())
        accuracies.append(correctness[mask].mean())
        counts.append(count)

    bin_centers = np.array(bin_centers)
    accuracies = np.array(accuracies)
    counts = np.array(counts)

    # ECE must be computed over ALL predictions with the same 15-bin
    # procedure as compute_metrics.py; min_bin_size only affects which
    # points are PLOTTED, never the reported number.
    ece = _ece_from_arrays(confidence, correctness, n_bins=n_bins)

    return bin_centers, accuracies, counts, ece


# ---------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------


def draw_reliability_panel(ax, df, model, language, color):
    """
    Draw one reliability-diagram panel onto `ax`, including:
      - diagonal (perfect calibration)
      - model curve with shaded over/under-confidence gap
      - inset bar histogram of sample count per bin
      - annotated ECE, accuracy, n
    """

    subset = df[(df["model"] == model) & (df["language"] == language)]

    confidence = (subset["confidence"] / 100).to_numpy()
    correctness = subset["is_correct"].astype(int).to_numpy()

    bin_centers, accuracies, counts, ece = compute_bins(confidence, correctness)

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2, zorder=1,
            label="Perfect calibration")

    if len(bin_centers) > 0:
        ax.fill_between(
            bin_centers, bin_centers, accuracies,
            where=(accuracies < bin_centers),
            color="#D55E00", alpha=0.18, interpolate=True, zorder=0,
        )
        ax.fill_between(
            bin_centers, bin_centers, accuracies,
            where=(accuracies >= bin_centers),
            color="#0072B2", alpha=0.18, interpolate=True, zorder=0,
        )

        ax.plot(bin_centers, accuracies, marker="o", markersize=4,
                 linewidth=1.8, color=color, zorder=3, label=model)

        inset = ax.inset_axes([0.62, 0.08, 0.34, 0.28])
        inset.bar(bin_centers, counts, width=0.05, color=color, alpha=0.6)
        inset.set_xlim(0, 1)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title("n / bin", fontsize=6, pad=2)
        for spine in inset.spines.values():
            spine.set_linewidth(0.5)

    overall_acc = correctness.mean() if len(correctness) else np.nan
    n_total = len(correctness)

    ax.text(
        0.03, 0.97,
        f"ECE={ece:.3f}\nAcc={overall_acc:.3f}\nn={n_total}",
        transform=ax.transAxes, ha="left", va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                   edgecolor="0.7", linewidth=0.5, alpha=0.9),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{model} \u2014 {language}", fontsize=9)


def reliability_diagrams_grid(df, output_dir, color_map):
    """
    One figure, subplot grid of languages (rows) x models (cols).
    """

    models = sorted(df["model"].unique())
    languages = sorted(df["language"].unique())

    n_rows, n_cols = len(languages), len(models)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.1 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for i, language in enumerate(languages):
        for j, model in enumerate(models):
            ax = axes[i][j]
            draw_reliability_panel(ax, df, model, language, color_map[model])
            if i == n_rows - 1:
                ax.set_xlabel("Confidence")
            if j == 0:
                ax.set_ylabel(f"{language}\nAccuracy")

    fig.suptitle("Reliability Diagrams by Model and Language", y=1.01, fontsize=12)
    fig.tight_layout()
    savefig_both(fig, output_dir, "reliability_diagrams_grid")
    plt.close(fig)


# ---------------------------------------------------------
# Confidence histogram
# ---------------------------------------------------------


def draw_confidence_panel(ax, df, model, language, color):

    subset = df[(df["model"] == model) & (df["language"] == language)]
    confidence = subset["confidence"].to_numpy()
    correctness = subset["is_correct"].astype(int).to_numpy()

    ax.hist(confidence, bins=10, range=(0, 100), color=color,
             edgecolor="black", linewidth=0.5, alpha=0.85)

    mean_conf = confidence.mean() if len(confidence) else np.nan
    mean_acc = correctness.mean() * 100 if len(correctness) else np.nan

    ax.axvline(mean_conf, color="#0072B2", linestyle="-", linewidth=1.3,
               label=f"Mean conf.\n({mean_conf:.1f})")
    ax.axvline(mean_acc, color="#D55E00", linestyle="--", linewidth=1.3,
               label=f"Accuracy\n({mean_acc:.1f})")

    ax.set_xlim(0, 100)
    ax.set_title(f"{model} \u2014 {language}", fontsize=9)
    ax.legend(fontsize=6, loc="upper left", framealpha=0.85)


def confidence_histograms_grid(df, output_dir, color_map):

    models = sorted(df["model"].unique())
    languages = sorted(df["language"].unique())

    n_rows, n_cols = len(languages), len(models)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.1 * n_cols, 2.6 * n_rows),
        squeeze=False,
    )

    for i, language in enumerate(languages):
        for j, model in enumerate(models):
            ax = axes[i][j]
            draw_confidence_panel(ax, df, model, language, color_map[model])
            if i == n_rows - 1:
                ax.set_xlabel("Confidence (%)")
            if j == 0:
                ax.set_ylabel(f"{language}\nCount")

    fig.suptitle("Confidence Distributions by Model and Language", y=1.01, fontsize=12)
    fig.tight_layout()
    savefig_both(fig, output_dir, "confidence_histograms_grid")
    plt.close(fig)


# ---------------------------------------------------------
# Individual (supplementary) figures
# ---------------------------------------------------------


def reliability_diagram(df, model, language, output_dir, color_map):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    draw_reliability_panel(ax, df, model, language, color_map[model])
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    savefig_both(fig, output_dir, f"{model}_{language}_reliability")
    plt.close(fig)


def confidence_histogram(df, model, language, output_dir, color_map):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    draw_confidence_panel(ax, df, model, language, color_map[model])
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    savefig_both(fig, output_dir, f"{model}_{language}_confidence_histogram")
    plt.close(fig)


# ---------------------------------------------------------
# Accuracy comparison (with bootstrap CI error bars, if available)
# ---------------------------------------------------------


def accuracy_barplot(metrics, output_dir, color_map):
    """Compare model accuracy using Overall metrics only."""

    overall = metrics[metrics["language"] == "Overall"].copy()
    overall = overall.sort_values("accuracy", ascending=False)

    colors = [color_map.get(m, "#888888") for m in overall["model"]]

    has_ci = {"accuracy_ci_lower", "accuracy_ci_upper"}.issubset(overall.columns)

    yerr = None
    if has_ci:
        lower_err = overall["accuracy"] - overall["accuracy_ci_lower"]
        upper_err = overall["accuracy_ci_upper"] - overall["accuracy"]
        yerr = [lower_err.to_numpy(), upper_err.to_numpy()]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(overall["model"], overall["accuracy"], color=colors,
                   edgecolor="black", linewidth=0.6,
                   yerr=yerr, capsize=4,
                   error_kw={"elinewidth": 1.2, "ecolor": "0.2"})

    label_offset = (overall["accuracy_ci_upper"] if has_ci else overall["accuracy"]).max() * 0.02

    for bar, val in zip(bars, overall["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + label_offset,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, min(1.08, (overall["accuracy_ci_upper"] if has_ci else overall["accuracy"]).max() * 1.15))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    title = "Overall Accuracy by Model"
    if has_ci:
        title += " (error bars: 95% bootstrap CI)"
    ax.set_title(title)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    savefig_both(fig, output_dir, "accuracy_comparison")
    plt.close(fig)


# ---------------------------------------------------------
# Expected Calibration Error comparison (with bootstrap CI error bars)
# ---------------------------------------------------------


def ece_barplot(metrics, output_dir, color_map):
    """Compare Expected Calibration Error across models (lower = better)."""

    overall = metrics[metrics["language"] == "Overall"].copy()
    overall = overall.sort_values("ece", ascending=True)

    colors = [color_map.get(m, "#888888") for m in overall["model"]]

    has_ci = {"ece_ci_lower", "ece_ci_upper"}.issubset(overall.columns)

    yerr = None
    if has_ci:
        lower_err = overall["ece"] - overall["ece_ci_lower"]
        upper_err = overall["ece_ci_upper"] - overall["ece"]
        yerr = [lower_err.to_numpy(), upper_err.to_numpy()]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(overall["model"], overall["ece"], color=colors,
                   edgecolor="black", linewidth=0.6,
                   yerr=yerr, capsize=4,
                   error_kw={"elinewidth": 1.2, "ecolor": "0.2"})

    label_offset = (overall["ece_ci_upper"] if has_ci else overall["ece"]).max() * 0.03

    for bar, val in zip(bars, overall["ece"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + label_offset,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("ECE (lower is better)")
    ax.set_xlabel("Model")
    title = "Expected Calibration Error by Model"
    if has_ci:
        title += " (error bars: 95% bootstrap CI)"
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    savefig_both(fig, output_dir, "ece_comparison")
    plt.close(fig)


# ---------------------------------------------------------
# Model x Language ECE heatmap
# ---------------------------------------------------------


def ece_heatmap(metrics, output_dir):
    """
    Heatmap of ECE across models (rows) and languages (cols), excluding
    the "Overall" pseudo-language. Shows calibration error for every
    model-language combination at a glance.
    """

    per_lang = metrics[metrics["language"] != "Overall"].copy()

    pivot = per_lang.pivot_table(
        index="model", columns="language", values="ece", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(1.1 * len(pivot.columns) + 2,
                                     0.6 * len(pivot.index) + 2))

    im = ax.imshow(pivot.values, cmap="Reds", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    vmax = np.nanmax(pivot.values)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val > 0.6 * vmax else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ECE")

    ax.set_title("Calibration Error (ECE) by Model and Language")
    ax.grid(False)

    fig.tight_layout()
    savefig_both(fig, output_dir, "ece_heatmap")
    plt.close(fig)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():

    config = load_config()

    metrics = load_metrics(config)
    predictions = load_predictions(config)

    output_dir = get_figure_directory(config)

    color_map = model_color_map(predictions["model"].unique())

    print("=" * 70)
    print("Generating figures...")
    print("=" * 70)

    reliability_diagrams_grid(predictions, output_dir, color_map)
    print("\u2713 Reliability diagrams (grid)")

    confidence_histograms_grid(predictions, output_dir, color_map)
    print("\u2713 Confidence histograms (grid)")

    accuracy_barplot(metrics, output_dir, color_map)
    print("\u2713 Accuracy comparison")

    ece_barplot(metrics, output_dir, color_map)
    print("\u2713 ECE comparison")

    ece_heatmap(metrics, output_dir)
    print("\u2713 ECE heatmap")

    if GENERATE_INDIVIDUAL:
        combinations = (
            predictions[["model", "language"]]
            .drop_duplicates()
            .sort_values(["model", "language"])
        )
        for _, row in combinations.iterrows():
            reliability_diagram(predictions, row["model"], row["language"],
                                  output_dir, color_map)
            confidence_histogram(predictions, row["model"], row["language"],
                                   output_dir, color_map)
        print("\u2713 Individual per-pair figures (supplementary)")

    print()
    print("=" * 70)
    print("Finished!")
    print(f"Figures saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
