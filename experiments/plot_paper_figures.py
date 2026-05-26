import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle


PLOT_ORDER = ["tan+rbf+mat", "tan+mat", "tan+rbf", "tan", "rbf+mat", "mat", "rbf"]

KERNEL_LABEL = "tan = Tanimoto Kernel   |   rbf = Radial Basis Function Kernel   |   mat = Matern Kernel"

COLORS = {
    "primary": "#0f9f94",
    "primary_dark": "#0b1328",
    "primary_mid": "#2faea5",
    "primary_light": "#b8d9d5",
    "blue": "#2f80ed",
    "slate": "#44546a",
    "grid": "#dfe7f1",
    "dash": "#c9d6e6",
    "row": "#eef3f8",
    "bg": "#f7fafc",
}


MODEL_F1 = pd.DataFrame(
    [
        ("Ours (Hybrid GP)", 0.803),
        ("LDA", 0.785),
        ("GP (Tanimoto)", 0.783),
        ("QDA", 0.774),
        ("GP (Standard RBF)", 0.758),
        ("Naive Bayes", 0.740),
    ],
    columns=["model", "F1"],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication figures used in results/paper_plot.")
    parser.add_argument(
        "--summary-csv",
        default=os.path.join("results", "comparison", "summary_metrics_with_tradeoff.csv"),
        help="Kernel summary CSV containing combo, AUC, F1, and Brier columns.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("results", "paper_plot"),
        help="Output directory for AUC.png, brier.png, f1.png, f1_model.png, and kernels.png.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_summary(path):
    df = pd.read_csv(path)
    required = {"combo", "AUC", "F1", "Brier"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    df = df[df["combo"].isin(PLOT_ORDER)].copy()
    df["combo"] = pd.Categorical(df["combo"], categories=PLOT_ORDER, ordered=True)
    return df.sort_values("combo").reset_index(drop=True)


def set_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "figure.facecolor": COLORS["bg"],
            "savefig.facecolor": COLORS["bg"],
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": COLORS["primary_dark"],
            "xtick.color": COLORS["slate"],
            "ytick.color": COLORS["slate"],
            "text.color": COLORS["primary_dark"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 1.0,
        }
    )


def metric_limits(values, higher_is_better):
    values = np.asarray(values, dtype=float)
    vmin = values.min()
    vmax = values.max()
    span = max(vmax - vmin, 1e-6)
    if higher_is_better:
        return vmin - span * 0.08, vmax + span * 0.16
    return vmin - span * 0.05, vmax + span * 0.16


def row_colors(df, best_idx):
    colors = []
    for idx, combo in enumerate(df["combo"].astype(str)):
        if idx == best_idx:
            colors.append(COLORS["primary"])
        elif combo in {"rbf+mat", "mat", "rbf"}:
            colors.append(COLORS["primary_light"])
        else:
            colors.append(COLORS["primary_mid"])
    return colors


def plot_metric_bar(df, metric, out_path, title, xlabel, higher_is_better=True, dpi=300):
    set_style()
    work = df.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)
    values = work[metric].astype(float).to_numpy()
    labels = work["combo"].astype(str).tolist()
    best_idx = 0
    colors = row_colors(work, best_idx)

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("white")

    y = np.arange(len(work))
    xmin, xmax = metric_limits(values, higher_is_better)

    for yi in y:
        ax.add_patch(
            Rectangle(
                (xmin, yi - 0.36),
                xmax - xmin,
                0.72,
                facecolor=COLORS["row"],
                edgecolor="none",
                zorder=0,
            )
        )

    bars = ax.barh(y, values, height=0.62, color=colors, edgecolor="none", zorder=3)
    bars[best_idx].set_edgecolor(COLORS["primary_dark"])
    bars[best_idx].set_linewidth(2.0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=17)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_xlim(xmin, xmax)
    ax.tick_params(axis="x", labelsize=16)
    ax.grid(axis="x", which="major", linestyle="-", alpha=0.9)
    ax.grid(axis="x", which="minor", linestyle=(0, (2, 5)), color=COLORS["dash"], linewidth=1.2)
    ax.xaxis.set_minor_locator(plt.MultipleLocator((xmax - xmin) / 6.0))

    for idx, (bar, val) in enumerate(zip(bars, values)):
        rank_label = f"#{idx + 1}"
        ax.text(
            xmin + (xmax - xmin) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            rank_label,
            ha="left",
            va="center",
            fontsize=14,
            color="white",
            zorder=4,
        )
        offset = (xmax - xmin) * 0.012
        weight = "bold" if idx == best_idx else "normal"
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            ha="left",
            va="center",
            fontsize=15,
            fontweight=weight,
            color=COLORS["primary_dark"] if idx == best_idx else COLORS["slate"],
            zorder=4,
        )

    fig.text(0.17, 0.93, title, fontsize=25, fontweight="bold", color=COLORS["primary_dark"])
    fig.text(0.17, 0.895, KERNEL_LABEL, fontsize=14, color=COLORS["slate"])
    plt.subplots_adjust(left=0.17, right=0.96, top=0.87, bottom=0.12)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def combo_color(combo):
    if combo in {"tan+rbf+mat", "tan+mat", "tan+rbf", "tan"}:
        return COLORS["primary"]
    if combo in {"rbf+mat", "mat"}:
        return COLORS["blue"]
    return COLORS["slate"]


def plot_kernel_tradeoff(df, out_path, dpi=300):
    set_style()
    work = df.copy()
    work["combo"] = pd.Categorical(work["combo"], categories=PLOT_ORDER, ordered=True)
    work = work.sort_values("combo").reset_index(drop=True)

    fig = plt.figure(figsize=(13.6, 7.4))
    fig.patch.set_facecolor(COLORS["bg"])
    ax = fig.add_axes([0.08, 0.12, 0.64, 0.76])
    legend_ax = fig.add_axes([0.73, 0.12, 0.25, 0.76])
    legend_ax.axis("off")

    x = work["Brier"].astype(float).to_numpy()
    y = work["F1"].astype(float).to_numpy()
    auc = work["AUC"].astype(float).to_numpy()
    combos = work["combo"].astype(str).tolist()

    ax.set_facecolor("white")
    ax.set_xlabel("Brier Score", fontsize=16)
    ax.set_ylabel("F1 Score", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.set_xlim(0.136, 0.242)
    ax.set_ylim(0.694, 0.818)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
    ax.grid(axis="both", which="major", color=COLORS["dash"], linestyle=(0, (2, 5)), linewidth=1.1)
    ax.grid(axis="both", which="minor", color=COLORS["grid"], linestyle="-", linewidth=0.8)

    ax.scatter([0.162], [0.794], s=9000, color="#dff7f2", alpha=0.55, edgecolor="none", zorder=0)
    ax.scatter([0.229], [0.729], s=10000, color="#eaf1f9", alpha=0.75, edgecolor="none", zorder=0)

    cmap = LinearSegmentedColormap.from_list("auc_map", ["#486d9e", "#2bb8db", "#05866f"])
    norm = Normalize(vmin=auc.min(), vmax=auc.max())

    right_x = 0.238
    right_y = np.linspace(0.793, 0.714, len(work))
    for xi, yi, ai, combo, ly in zip(x, y, auc, combos, right_y):
        color = combo_color(combo)
        ax.plot([xi, right_x], [yi, ly], color=color, alpha=0.55, linewidth=1.2, zorder=1)
        ax.scatter([xi], [yi], s=250, color=color, edgecolor="#e7f5f3", linewidth=5, zorder=4)
        ax.scatter([xi], [yi], s=55, color="white", edgecolor="none", zorder=5)

    cax = fig.add_axes([0.095, 0.15, 0.18, 0.03])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    cax.imshow(gradient, aspect="auto", cmap=cmap)
    cax.set_xticks([0, 128, 255])
    cax.set_xticklabels([f"{auc.min():.2f}", "0.84", f"{auc.max():.2f}"], fontsize=9, color=COLORS["slate"])
    cax.set_yticks([])
    cax.text(1.02, 1.1, "AUC", transform=cax.transAxes, fontsize=10, color=COLORS["slate"])
    for spine in cax.spines.values():
        spine.set_visible(False)

    card = FancyBboxPatch(
        (0.02, 0.04),
        0.94,
        0.90,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        transform=legend_ax.transAxes,
        facecolor="white",
        edgecolor="#d9e2ee",
        linewidth=1.2,
    )
    legend_ax.add_patch(card)
    legend_ax.text(0.10, 0.90, "Kernel Family", transform=legend_ax.transAxes, fontsize=15, fontweight="bold")

    y0 = 0.79
    dy = 0.105
    for idx, row in work.iterrows():
        combo = str(row["combo"])
        color = combo_color(combo)
        yy = y0 - idx * dy
        legend_ax.scatter([0.11], [yy], s=220, color=color, transform=legend_ax.transAxes, zorder=5)
        legend_ax.scatter([0.11], [yy], s=35, color="white", transform=legend_ax.transAxes, zorder=6)
        legend_ax.text(0.19, yy + 0.018, combo.replace("+", " + "), transform=legend_ax.transAxes, fontsize=13, fontweight="bold")
        legend_ax.text(
            0.19,
            yy - 0.035,
            f"F1 {row['F1']:.3f}   Brier {row['Brier']:.3f}   AUC {row['AUC']:.3f}",
            transform=legend_ax.transAxes,
            fontsize=10.5,
            color=COLORS["slate"],
        )
        if idx < len(work) - 1:
            legend_ax.plot([0.10, 0.90], [yy - 0.058, yy - 0.058], color="#e8eef5", transform=legend_ax.transAxes, linewidth=1.0)

    fig.text(0.08, 0.94, "Kernel Trade-off Plot", fontsize=20, fontweight="bold", color=COLORS["primary_dark"])
    fig.text(0.08, 0.91, KERNEL_LABEL, fontsize=11.5, color=COLORS["slate"])
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_model_f1(out_path, dpi=300):
    set_style()
    work = MODEL_F1.sort_values("F1", ascending=True)
    fig, ax = plt.subplots(figsize=(7.1, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = ["#c73a3a"] * len(work)
    colors[work["model"].tolist().index("Ours (Hybrid GP)")] = "#3c9638"
    bars = ax.barh(work["model"], work["F1"], color=colors, edgecolor="black", linewidth=0.8)

    ax.set_title("F1 Score Comparison", fontsize=12, fontweight="bold")
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Model")
    ax.set_xlim(0.65, 0.85)
    ax.grid(axis="x", linestyle="-", alpha=0.45)
    ax.grid(axis="y", visible=False)
    ax.tick_params(labelsize=9)

    for bar in bars:
        val = bar.get_width()
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", ha="left", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    df = load_summary(args.summary_csv)

    plot_metric_bar(df, "AUC", os.path.join(args.out_dir, "AUC.png"), "Kernel Comparison: AUC", "AUC", True, args.dpi)
    plot_metric_bar(df, "Brier", os.path.join(args.out_dir, "brier.png"), "Kernel Comparison: Brier", "Brier", False, args.dpi)
    plot_metric_bar(df, "F1", os.path.join(args.out_dir, "f1.png"), "Kernel Comparison: F1", "F1", True, args.dpi)
    plot_kernel_tradeoff(df, os.path.join(args.out_dir, "kernels.png"), args.dpi)
    plot_model_f1(os.path.join(args.out_dir, "f1_model.png"), args.dpi)

    print(f"[Saved] paper figures under {os.path.abspath(args.out_dir)}")
    print("[Note] arch.png is intentionally not generated by this script.")


if __name__ == "__main__":
    main()
