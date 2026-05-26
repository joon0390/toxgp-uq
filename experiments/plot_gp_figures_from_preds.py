import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GP diagnostic figures from saved prediction CSV.")
    parser.add_argument("--preds-csv", required=True)
    parser.add_argument("--out-prefix", default=None)
    return parser.parse_args()


def resolve_prefix(preds_csv, out_prefix):
    if out_prefix:
        return out_prefix
    path = os.path.abspath(preds_csv)
    if path.lower().endswith("_preds.csv"):
        return path[:-10]
    return os.path.splitext(path)[0]


def plot_roc(y_true, proba, out_prefix):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_value = roc_auc_score(y_true, proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc.png", dpi=300)
    plt.close()


def plot_pr(y_true, proba, out_prefix):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pr.png", dpi=300)
    plt.close()


def plot_confusion(y_true, y_pred, out_prefix):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (row, col), value in np.ndenumerate(cm):
        plt.text(col, row, f"{value}", ha="center", va="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_confusion.png", dpi=300)
    plt.close()


def plot_score_hist(y_true, proba, out_prefix):
    plt.figure(figsize=(6, 4))
    plt.hist(proba[y_true == 0], bins=30, alpha=0.65, label="Class 0")
    plt.hist(proba[y_true == 1], bins=30, alpha=0.65, label="Class 1")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Prediction Score Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_score_hist.png", dpi=300)
    plt.close()


def plot_calibration(y_true, proba, out_prefix):
    prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10)
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(prob_pred, prob_true, "o-", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title("Reliability Curve")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_calibration.png", dpi=300)
    plt.close()


def main():
    args = parse_args()
    preds = pd.read_csv(args.preds_csv)
    out_prefix = resolve_prefix(args.preds_csv, args.out_prefix)

    y_true = preds["y_true"].astype(int).to_numpy()
    y_pred = preds["y_pred"].astype(int).to_numpy()
    proba = preds["p_hat"].astype(float).to_numpy()

    plot_roc(y_true, proba, out_prefix)
    plot_pr(y_true, proba, out_prefix)
    plot_confusion(y_true, y_pred, out_prefix)
    plot_score_hist(y_true, proba, out_prefix)
    plot_calibration(y_true, proba, out_prefix)

    print(f"[Saved] {out_prefix}_roc.png")
    print(f"[Saved] {out_prefix}_pr.png")
    print(f"[Saved] {out_prefix}_confusion.png")
    print(f"[Saved] {out_prefix}_score_hist.png")
    print(f"[Saved] {out_prefix}_calibration.png")


if __name__ == "__main__":
    main()
