"""
Apply Benjamini-Hochberg FDR correction to the pairwise significance
table produced by compute_metrics.py.

Why this matters: pairwise_significance.csv contains many hypothesis
tests (10 language pairs x 2 metrics x 3 models = 60 tests). At an
uncorrected alpha=0.05, roughly 3 false positives are expected by
chance alone even if there were no real effects anywhere. This script
adds Benjamini-Hochberg adjusted p-values (q-values) and corresponding
significance flags, so both raw and FDR-corrected results can be
reported alongside each other.

Two corrections are added:
  - p_value_fdr_by_metric / significant_fdr_by_metric:
        BH correction applied SEPARATELY within the accuracy family
        (30 tests) and the ece family (30 tests). This is the
        recommended primary result to report, since accuracy and ECE
        answer different scientific questions and are conventionally
        treated as separate hypothesis families.
  - p_value_fdr_all / significant_fdr_all:
        BH correction applied across ALL 60 tests together, as a
        stricter robustness check.

This does NOT re-run any bootstrap resampling -- it only re-processes
the p-values already saved in pairwise_significance.csv, so it runs in
seconds regardless of how long the original bootstrap took.

Usage
-----
    python -m src.apply_fdr_correction
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def load_config():
    """
    Load experiment configuration.
    """

    config_path = Path("config/config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def benjamini_hochberg(pvalues):
    """
    Compute Benjamini-Hochberg FDR-adjusted p-values (q-values).

    Parameters
    ----------
    pvalues : array-like of float
        Raw p-values from a family of hypothesis tests.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values, same order as input, each clipped to [0, 1].
    """

    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)

    if n == 0:
        return np.array([])

    order = np.argsort(pvalues)
    ranked = pvalues[order]

    adjusted_ranked = np.empty(n)
    running_min = 1.0

    # Walk from the largest p-value down to the smallest, taking a
    # running minimum -- this is the standard BH step-up procedure.
    for i in range(n - 1, -1, -1):
        rank = i + 1
        candidate = ranked[i] * n / rank
        running_min = min(running_min, candidate)
        adjusted_ranked[i] = running_min

    adjusted = np.empty(n)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)

    return adjusted


def apply_corrections(pairwise, alpha=0.05):
    """
    Add FDR-corrected p-values to the pairwise significance table.
    """

    pairwise = pairwise.copy()

    # Correction within each metric family (recommended primary reporting).
    pairwise["p_value_fdr_by_metric"] = np.nan

    for metric_name, group in pairwise.groupby("metric"):
        adjusted = benjamini_hochberg(group["p_value"].to_numpy())
        pairwise.loc[group.index, "p_value_fdr_by_metric"] = adjusted

    pairwise["significant_fdr_by_metric"] = (
        pairwise["p_value_fdr_by_metric"] < alpha
    )

    # Correction across the full table (stricter robustness check).
    pairwise["p_value_fdr_all"] = benjamini_hochberg(
        pairwise["p_value"].to_numpy()
    )
    pairwise["significant_fdr_all"] = pairwise["p_value_fdr_all"] < alpha

    pairwise = pairwise.round(
        {
            "p_value_fdr_by_metric": 4,
            "p_value_fdr_all": 4,
        }
    )

    return pairwise


def main():

    config = load_config()

    metrics_path = Path(config["paths"]["processed_results"])
    pairwise_path = metrics_path.parent / "pairwise_significance.csv"

    if not pairwise_path.exists():
        raise FileNotFoundError(
            f"{pairwise_path} not found. Run `python -m src.compute_metrics` first."
        )

    pairwise = pd.read_csv(pairwise_path)

    corrected = apply_corrections(pairwise, alpha=0.05)

    corrected.to_csv(pairwise_path, index=False)

    print("=" * 70)
    print("Added FDR-corrected p-values to pairwise_significance.csv")
    print("=" * 70)
    print()
    print(corrected.to_string(index=False))
    print()
    print(f"Saved to: {pairwise_path}")


if __name__ == "__main__":
    main()