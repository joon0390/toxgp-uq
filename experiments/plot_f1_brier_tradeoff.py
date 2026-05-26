import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot F1 vs Brier trade-off with best trade-off and preferred region.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--override-combo", default=None)
    parser.add_argument("--override-metrics-txt", default=None)
    return parser.parse_args()


def load_metrics_txt(path):
    values = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                values[key] = float(value)
            except ValueError:
                continue
    return values


def choose_best_tradeoff(df):
    f1_min, f1_max = df["F1"].min(), df["F1"].max()
    brier_min, brier_max = df["Brier"].min(), df["Brier"].max()

    if f1_max > f1_min:
        f1_norm = (df["F1"] - f1_min) / (f1_max - f1_min)
    else:
        f1_norm = pd.Series(np.ones(len(df)), index=df.index)

    if brier_max > brier_min:
        brier_norm = (df["Brier"] - brier_min) / (brier_max - brier_min)
    else:
        brier_norm = pd.Series(np.zeros(len(df)), index=df.index)

    df = df.copy()
    df["tradeoff_score"] = f1_norm + (1.0 - brier_norm)
    best_idx = df["tradeoff_score"].idxmax()
    return df, best_idx


def main():
    args = parse_args()
    df = pd.read_csv(args.summary_csv)

    if args.override_combo and args.override_metrics_txt:
        metrics = load_metrics_txt(args.override_metrics_txt)
        mask = df["combo"] == args.override_combo
        if mask.any():
            if "F1" in metrics:
                df.loc[mask, "F1"] = metrics["F1"]
            if "Brier" in metrics:
                df.loc[mask, "Brier"] = metrics["Brier"]
            if "AUC" in metrics and "AUC" in df.columns:
                df.loc[mask, "AUC"] = metrics["AUC"]
            if "ACC" in metrics and "ACC" in df.columns:
                df.loc[mask, "ACC"] = metrics["ACC"]
        else:
            raise ValueError(f"Combo '{args.override_combo}' not found in {args.summary_csv}")

    df, best_idx = choose_best_tradeoff(df)

    x_min, x_max = df["Brier"].min(), df["Brier"].max()
    y_min, y_max = df["F1"].min(), df["F1"].max()
    x_pad = max((x_max - x_min) * 0.08, 0.005)
    y_pad = max((y_max - y_min) * 0.08, 0.005)

    pref_x = x_min + (x_max - x_min) * 0.30
    pref_y = y_max - (y_max - y_min) * 0.30

    plt.figure(figsize=(7, 6))
    plt.axvspan(x_min - x_pad, pref_x, ymin=(pref_y - (y_min - y_pad)) / ((y_max + y_pad) - (y_min - y_pad)), ymax=1.0,
                color="#d1fae5", alpha=0.45, label="Preferred region")

    for idx, row in df.iterrows():
        color = "#0f766e" if idx == best_idx else "#2563eb"
        size = 120 if idx == best_idx else 80
        plt.scatter(row["Brier"], row["F1"], s=size, color=color, edgecolor="black", linewidth=0.6, zorder=3)
        label = str(row["combo"])
        if idx == best_idx:
            label = f"{label} (best trade-off)"
        plt.annotate(label, (row["Brier"], row["F1"]), xytext=(6, 6), textcoords="offset points", fontsize=9)

    best_row = df.loc[best_idx]
    plt.annotate(
        "Preferred region: upper-left",
        xy=(pref_x, pref_y),
        xytext=(pref_x + 0.008, pref_y - 0.01),
        arrowprops={"arrowstyle": "->", "color": "#065f46", "lw": 1.2},
        fontsize=10,
        color="#065f46",
    )
    plt.annotate(
        f"Best trade-off: {best_row['combo']}",
        xy=(best_row["Brier"], best_row["F1"]),
        xytext=(best_row["Brier"] + 0.01, best_row["F1"] - 0.01),
        arrowprops={"arrowstyle": "->", "color": "#111827", "lw": 1.2},
        fontsize=10,
        color="#111827",
    )

    plt.xlabel("Brier Score (lower is better)")
    plt.ylabel("F1 Score (higher is better)")
    plt.title("F1 Score vs Brier Score")
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    plt.close()

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)

    print(f"[Saved] {os.path.abspath(args.out_png)}")
    if args.out_csv:
        print(f"[Saved] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
