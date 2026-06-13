"""
Calibration metrics and plots.

The key question: when a model says it is X% confident, is it right about X% of
the time? These functions measure that and turn the raw results table into a
summary you can put in your paper.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # lets us save figures without a screen; comment out in a notebook
import matplotlib.pyplot as plt


def accuracy(is_correct):
    return float(np.mean(is_correct))


def brier_score(confidence, is_correct):
    """Mean squared gap between confidence and correctness. Lower is better."""
    confidence = np.asarray(confidence, dtype=float)
    is_correct = np.asarray(is_correct, dtype=float)
    return float(np.mean((confidence - is_correct) ** 2))


def expected_calibration_error(confidence, is_correct, n_bins=10):
    """
    ECE: split predictions into confidence bins, and in each bin compare the
    average confidence to the actual accuracy. The weighted average of those gaps
    is the ECE. 0 = perfectly calibrated; lower is better.
    """
    confidence = np.asarray(confidence, dtype=float)
    is_correct = np.asarray(is_correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidence)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence > lo) & (confidence <= hi)
        count = int(mask.sum())
        if count > 0:
            bin_acc = is_correct[mask].mean()
            bin_conf = confidence[mask].mean()
            ece += (count / total) * abs(bin_acc - bin_conf)
    return float(ece)


def overconfidence(confidence, is_correct):
    """Positive = the model is on average more confident than it is accurate."""
    return float(np.mean(confidence) - np.mean(is_correct))


def selective_auroc(confidence, is_correct):
    """
    How well confidence separates right answers from wrong ones (1.0 = perfect,
    0.5 = no better than chance). Useful for 'should the model abstain?' questions.
    """
    from sklearn.metrics import roc_auc_score
    is_correct = np.asarray(is_correct)
    if len(np.unique(is_correct)) < 2:
        return float("nan")
    return float(roc_auc_score(is_correct, confidence))


def reliability_diagram(confidence, is_correct, n_bins=10, title="", save_path=None):
    """Plot accuracy vs confidence per bin. The closer to the diagonal, the better."""
    confidence = np.asarray(confidence, dtype=float)
    is_correct = np.asarray(is_correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() > 0:
            centers.append((lo + hi) / 2)
            accs.append(is_correct[mask].mean())

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    ax.plot(centers, accs, marker="o", label="model")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def summarize(df):
    """
    Given the results table, compute one row of metrics per
    (model, method, language). Rows where the answer or confidence could not be
    parsed are dropped first.
    """
    import pandas as pd
    clean = df.dropna(subset=["is_correct", "confidence"]).copy()
    out = []
    group_cols = ["model_id", "method", "language_name"]
    for keys, g in clean.groupby(group_cols):
        conf = g["confidence"].to_numpy()
        correct = g["is_correct"].to_numpy()
        out.append({
            "model_id": keys[0],
            "method": keys[1],
            "language": keys[2],
            "n": len(g),
            "accuracy": round(accuracy(correct), 3),
            "ece": round(expected_calibration_error(conf, correct), 3),
            "brier": round(brier_score(conf, correct), 3),
            "overconfidence": round(overconfidence(conf, correct), 3),
            "selective_auroc": round(selective_auroc(conf, correct), 3),
        })
    # Note: the output uses "language" as the column name (not "language_name").
    return pd.DataFrame(out).sort_values(["model_id", "method", "language"]).reset_index(drop=True)
