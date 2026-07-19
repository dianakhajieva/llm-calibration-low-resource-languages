"""
Compute calibration metrics for multilingual LLM evaluation, using
PAIRED bootstrap significance testing between languages.

Why paired: every language in this project is answered on the SAME
underlying 500 aligned questions (per model). Treating each language's
results as an independent sample throws away that structure and wastes
statistical power -- some questions are just harder than others,
regardless of language, and that per-question difficulty is noise we
can cancel out by pairing on question_id before resampling.

Outputs
-------
  1. metrics.csv
        Accuracy, average confidence, Brier score, and ECE per
        model-language pair (+ 95% bootstrap CIs on accuracy and ECE).

  2. pairwise_significance.csv
        For every pair of languages WITHIN each model, a PAIRED
        bootstrap test (resampling question indices, not row indices
        independently) of whether the difference in accuracy / ECE
        between the two languages is distinguishable from zero.

"""

from itertools import combinations
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

    return _ece_from_arrays(confidence, correctness, n_bins=n_bins)


def _ece_from_arrays(confidence, correctness, n_bins=15):
    """
    Same ECE computation as expected_calibration_error(), but operating
    directly on numpy arrays. Used by both the main metric computation
    and the bootstrap resampling below, so the two always agree.
    """

    bin_edges = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    ece = 0.0
    total = len(confidence)

    if total == 0:
        return float("nan")

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

        weight = np.sum(mask) / total

        ece += weight * abs(
            bin_accuracy - bin_confidence
        )

    return ece


# Metric functions available for bootstrap CIs and pairwise tests, keyed
# by the name used in metrics.csv / pairwise_significance.csv.
METRIC_FUNCTIONS = {
    "accuracy": accuracy,
    "ece": expected_calibration_error,
}


# ---------------------------------------------------------------------
# Bootstrap Confidence Intervals (single group, unpaired -- used for
# the per-language rows in metrics.csv, where there's only one group)
# ---------------------------------------------------------------------


def bootstrap_ci(
    df,
    metric_fn,
    n_bootstrap=1000,
    ci=0.95,
    seed=42,
):
    """
    Generic bootstrap CI for a metric computed on a group's rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows for one (model, language) group.

    metric_fn : callable
        Function taking a DataFrame and returning a float
        (e.g. accuracy, expected_calibration_error).

    n_bootstrap : int
        Number of resamples.

    ci : float
        Confidence level (0.95 = 95% CI).

    seed : int
        Random seed, for reproducibility.

    Returns
    -------
    (float, float)
        (lower_bound, upper_bound) of the confidence interval.
    """

    rng = np.random.default_rng(seed)

    n = len(df)

    if n == 0:
        return float("nan"), float("nan")

    values = np.empty(n_bootstrap)

    for i in range(n_bootstrap):

        sample_idx = rng.integers(0, n, size=n)
        resampled = df.iloc[sample_idx]
        values[i] = metric_fn(resampled)

    alpha = 1 - ci
    lower = np.nanpercentile(values, 100 * (alpha / 2))
    upper = np.nanpercentile(values, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


# ---------------------------------------------------------------------
# PAIRED bootstrap significance test (two languages, same questions)
# ---------------------------------------------------------------------


def paired_bootstrap_diff_test(
    df_a,
    df_b,
    metric_fn,
    n_bootstrap=1000,
    ci=0.95,
    seed=42,
    id_col="question_id",
):
    """
    PAIRED bootstrap test of whether metric_fn(lang_a) - metric_fn(lang_b)
    is distinguishable from zero, exploiting the fact that both
    languages were evaluated on the SAME underlying questions.

    Mechanics
    ---------
    1. Inner-join df_a and df_b on `id_col`, so we only keep questions
       both languages actually have a response for (drops any question
       that was skipped/failed in either language).
    2. Each bootstrap iteration resamples QUESTION indices (not row
       indices independently) -- the same resampled question goes into
       both the language-A and language-B computation for that
       iteration. This cancels out per-question difficulty as a source
       of noise, which an unpaired/independent bootstrap cannot do.
    3. Report the point difference, a 95% CI on the difference, and a
       two-sided bootstrap p-value.

    Returns
    -------
    dict with keys: diff, ci_lower, ci_upper, p_value, significant, n_paired
    """

    merged = df_a.merge(df_b, on=id_col, suffixes=("_a", "_b"))

    n = len(merged)

    if n == 0:
        return {
            "diff": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "n_paired": 0,
        }

    # Rebuild two small DataFrames with the plain column names that
    # metric_fn expects ("confidence", "is_correct"), so the same
    # metric functions work unchanged on paired data.
    cols_a = pd.DataFrame(
        {
            "confidence": merged["confidence_a"],
            "is_correct": merged["is_correct_a"],
        }
    )
    cols_b = pd.DataFrame(
        {
            "confidence": merged["confidence_b"],
            "is_correct": merged["is_correct_b"],
        }
    )

    rng = np.random.default_rng(seed)

    diffs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):

        # Same resampled question indices used for BOTH languages --
        # this is what makes it "paired".
        idx = rng.integers(0, n, size=n)

        resampled_a = cols_a.iloc[idx]
        resampled_b = cols_b.iloc[idx]

        diffs[i] = metric_fn(resampled_a) - metric_fn(resampled_b)

    alpha = 1 - ci
    ci_lower = np.nanpercentile(diffs, 100 * (alpha / 2))
    ci_upper = np.nanpercentile(diffs, 100 * (1 - alpha / 2))

    point_diff = metric_fn(cols_a) - metric_fn(cols_b)

    frac_le_zero = np.mean(diffs <= 0)
    frac_ge_zero = np.mean(diffs >= 0)
    p_value = min(1.0, 2 * min(frac_le_zero, frac_ge_zero))

    return {
        "diff": float(point_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "significant": bool(p_value < (1 - ci)),
        "n_paired": n,
    }


def compute_pairwise_significance(
    df,
    metric_names=("accuracy", "ece"),
    n_bootstrap=1000,
    id_col="question_id",
):
    """
    For every model, and every pair of languages that model was
    evaluated on (excluding "Overall"), run a PAIRED bootstrap
    difference test for each metric in metric_names.

    Returns
    -------
    pandas.DataFrame with one row per (model, language_a, language_b,
    metric) combination.
    """

    rows = []

    for model, model_df in df.groupby("model", sort=True):

        languages = sorted(model_df["language"].unique())

        for lang_a, lang_b in combinations(languages, 2):

            group_a = model_df[model_df["language"] == lang_a]
            group_b = model_df[model_df["language"] == lang_b]

            for metric_name in metric_names:

                metric_fn = METRIC_FUNCTIONS[metric_name]

                result = paired_bootstrap_diff_test(
                    group_a,
                    group_b,
                    metric_fn,
                    n_bootstrap=n_bootstrap,
                    id_col=id_col,
                )

                rows.append(
                    {
                        "model": model,
                        "language_a": lang_a,
                        "language_b": lang_b,
                        "metric": metric_name,
                        "n_paired_questions": result["n_paired"],
                        "value_a": round(metric_fn(group_a), 4),
                        "value_b": round(metric_fn(group_b), 4),
                        "diff_a_minus_b": round(result["diff"], 4),
                        "diff_ci_lower": round(result["ci_lower"], 4),
                        "diff_ci_upper": round(result["ci_upper"], 4),
                        "p_value": round(result["p_value"], 4),
                        "significant_at_0.05": result["significant"],
                    }
                )

    pairwise = pd.DataFrame(rows)

    if not pairwise.empty:
        pairwise = pairwise.sort_values(
            ["model", "metric", "language_a", "language_b"]
        ).reset_index(drop=True)

    return pairwise


# ---------------------------------------------------------------------
# Compute Metrics
# ---------------------------------------------------------------------


def compute_metrics(df, config, n_bootstrap=1000):
    """
    Compute metrics for every model-language pair
    and an overall summary for each model, including
    95% bootstrap confidence intervals for accuracy and ECE.
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

        acc_lo, acc_hi = bootstrap_ci(
            group, accuracy, n_bootstrap=n_bootstrap,
        )
        ece_lo, ece_hi = bootstrap_ci(
            group, expected_calibration_error, n_bootstrap=n_bootstrap,
        )

        rows.append(
            {
                "model": model,
                "language": language,
                "answered_questions": answered,
                "expected_questions": expected_per_language,
                "skipped_questions": expected_per_language - answered,
                "accuracy": accuracy(group),
                "accuracy_ci_lower": acc_lo,
                "accuracy_ci_upper": acc_hi,
                "average_confidence": average_confidence(group),
                "brier_score": brier_score(group),
                "ece": expected_calibration_error(group),
                "ece_ci_lower": ece_lo,
                "ece_ci_upper": ece_hi,
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

        acc_lo, acc_hi = bootstrap_ci(
            group, accuracy, n_bootstrap=n_bootstrap,
        )
        ece_lo, ece_hi = bootstrap_ci(
            group, expected_calibration_error, n_bootstrap=n_bootstrap,
        )

        rows.append(
            {
                "model": model,
                "language": "Overall",
                "answered_questions": answered,
                "expected_questions": expected_overall,
                "skipped_questions": expected_overall - answered,
                "accuracy": accuracy(group),
                "accuracy_ci_lower": acc_lo,
                "accuracy_ci_upper": acc_hi,
                "average_confidence": average_confidence(group),
                "brier_score": brier_score(group),
                "ece": expected_calibration_error(group),
                "ece_ci_lower": ece_lo,
                "ece_ci_upper": ece_hi,
            }
        )

    metrics = pd.DataFrame(rows)

    metrics = metrics.sort_values(
        ["model", "language"]
    )

    metrics = metrics.round(
        {
            "accuracy": 4,
            "accuracy_ci_lower": 4,
            "accuracy_ci_upper": 4,
            "average_confidence": 4,
            "brier_score": 4,
            "ece": 4,
            "ece_ci_lower": 4,
            "ece_ci_upper": 4,
        }
    )

    return metrics.reset_index(drop=True)


# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------


def save_metrics(metrics, config):
    """
    Save the main metrics dataframe.
    """

    output_path = Path(
        config["paths"]["processed_results"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_pairwise_significance(pairwise, config):
    """
    Save the pairwise significance dataframe alongside metrics.csv.
    """

    metrics_path = Path(config["paths"]["processed_results"])

    output_path = metrics_path.parent / "pairwise_significance.csv"

    pairwise.to_csv(output_path, index=False)

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
        n_bootstrap=1000,
    )

    metrics_path = save_metrics(metrics, config)

    print("=" * 70)
    print("Calibration Metrics (with 95% bootstrap CIs)")
    print("=" * 70)
    print()
    print(metrics)
    print()
    print(f"Saved metrics to: {metrics_path}")

    pairwise = compute_pairwise_significance(
        predictions[predictions["language"] != "Overall"]
        if "Overall" in predictions["language"].unique()
        else predictions,
        metric_names=("accuracy", "ece"),
        n_bootstrap=1000,
    )

    pairwise_path = save_pairwise_significance(pairwise, config)

    print()
    print("=" * 70)
    print("Paired Bootstrap Pairwise Language Significance Tests (per model)")
    print("=" * 70)
    print()
    print(pairwise.to_string(index=False))
    print()
    print(f"Saved pairwise significance table to: {pairwise_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()