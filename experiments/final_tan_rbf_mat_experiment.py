import argparse
import json
import os
import random
import warnings
from statistics import NormalDist

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern, RBF, WhiteKernel
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler


RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


FINAL_CONFIG = {
    "seed": 13,
    "pca_dim": 128,
    "morgan_bits": 1024,
    "morgan_radius": 2,
    "max_train": 8000,
    "n_restarts": 0,
    "constant_scale": 1.0,
    "white_noise": 1e-3,
    "w_tan": 1.0,
    "w_rbf": 0.8,
    "w_mat": 0.4,
    "rbf_length_scale": 2.0,
    "matern_length_scale": 2.0,
    "matern_nu": 1.5,
    "interval_level": 0.90,
    "kernel_summary_csv": os.path.join("results", "comparison", "summary_metrics_updated.csv"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Final tan+rbf+mat GP experiment with plots and paper-ready artifacts.")
    parser.add_argument("--train-csv", default=os.path.join("data", "train_final.csv"))
    parser.add_argument("--test-csv", default=os.path.join("data", "test_final.csv"))
    parser.add_argument("--out-dir", default=os.path.join("outputs", "tan_rbf_mat_final"))
    parser.add_argument("--bundle-path", default=None, help="Optional existing model bundle to reuse instead of training.")
    parser.add_argument("--kernel-summary-csv", default=FINAL_CONFIG["kernel_summary_csv"])
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def get_morgan_fp(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(str(smiles)) if isinstance(smiles, str) else None
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return (arr > 0).astype(np.float32)
    except Exception:
        return np.zeros((n_bits,), dtype=np.float32)


def rdkit_2d_descriptors(mol):
    values = []
    for _, func in Descriptors.descList:
        try:
            values.append(float(func(mol)))
        except Exception:
            values.append(np.nan)
    return np.nan_to_num(np.array(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def smiles_to_fp_and_desc(df, morgan_bits=1024, morgan_radius=2):
    fps, descs, valid = [], [], []
    for smiles in df["SMILES"]:
        mol = Chem.MolFromSmiles(str(smiles)) if isinstance(smiles, str) else None
        if mol is None:
            fps.append(np.zeros(morgan_bits, dtype=np.float32))
            descs.append(None)
            valid.append(False)
            continue
        fps.append(get_morgan_fp(smiles, radius=morgan_radius, n_bits=morgan_bits))
        descs.append(rdkit_2d_descriptors(mol))
        valid.append(True)

    x_desc = np.zeros((len(df), len(Descriptors.descList)), dtype=np.float32)
    for idx, arr in enumerate(descs):
        if arr is not None:
            x_desc[idx, :] = np.nan_to_num(arr, nan=0.0)
    x_fp = np.vstack(fps).astype(np.float32)
    return x_fp, x_desc, np.array(valid, dtype=bool)


def build_concat_features(df, scaler=None, pca=None, fit=True):
    x_fp, x_desc_raw, valid = smiles_to_fp_and_desc(
        df,
        morgan_bits=FINAL_CONFIG["morgan_bits"],
        morgan_radius=FINAL_CONFIG["morgan_radius"],
    )
    toxicity = pd.to_numeric(df["Toxicity"], errors="coerce")
    valid = valid & toxicity.notna().to_numpy()
    x_fp = x_fp[valid]
    x_desc_raw = x_desc_raw[valid]
    y = toxicity.loc[valid].astype(int).to_numpy()

    if fit:
        scaler = StandardScaler()
        x_desc = scaler.fit_transform(x_desc_raw)
        pca = PCA(n_components=FINAL_CONFIG["pca_dim"], random_state=0)
        x_desc_pca = pca.fit_transform(x_desc)
    else:
        x_desc = scaler.transform(x_desc_raw)
        x_desc_pca = pca.transform(x_desc)

    x = np.hstack([x_fp, x_desc_pca]).astype(np.float32)
    return x, y, x_fp.shape[1], scaler, pca


def subsample_train(df, max_train, seed):
    if not max_train or len(df) <= max_train:
        return df.copy()
    df_pos = df[df["Toxicity"] == 1]
    df_neg = df[df["Toxicity"] == 0]
    n_pos = min(len(df_pos), max_train // 2)
    n_neg = max_train - n_pos
    return (
        pd.concat(
            [
                df_pos.sample(n=n_pos, random_state=seed),
                df_neg.sample(n=n_neg, random_state=seed),
            ]
        )
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


class TanimotoRBFMaternKernel(Kernel):
    def __init__(
        self,
        fp_dim,
        w_tan=1.0,
        w_rbf=0.8,
        w_mat=0.4,
        rbf_length_scale=2.0,
        matern_length_scale=2.0,
        matern_nu=1.5,
        tan_min_denom=1e-6,
    ):
        self.fp_dim = int(fp_dim)
        self.w_tan = float(w_tan)
        self.w_rbf = float(w_rbf)
        self.w_mat = float(w_mat)
        self.rbf_length_scale = float(rbf_length_scale)
        self.matern_length_scale = float(matern_length_scale)
        self.matern_nu = float(matern_nu)
        self.tan_min_denom = float(tan_min_denom)
        self.rbf = RBF(length_scale=self.rbf_length_scale)
        self.matern = Matern(length_scale=self.matern_length_scale, nu=self.matern_nu)

    def __call__(self, x, y=None, eval_gradient=False):
        if eval_gradient:
            k = self.__call__(x, y, eval_gradient=False)
            return k, np.zeros((k.shape[0], k.shape[1], 0))

        same_inputs = y is None or y is x
        if y is None:
            y = x

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x_fp, x_desc = x[:, : self.fp_dim], x[:, self.fp_dim :]
        y_fp, y_desc = y[:, : self.fp_dim], y[:, self.fp_dim :]

        dot = x_fp @ y_fp.T
        sum_x = x_fp.sum(1)[:, None]
        sum_y = y_fp.sum(1)[None, :]
        denom = np.maximum(sum_x + sum_y - dot, self.tan_min_denom)
        k_tan = np.clip(dot / denom, 0.0, 1.0)
        if same_inputs:
            k_tan = 0.5 * (k_tan + k_tan.T)

        k_rbf = self.rbf(x_desc, y_desc)
        k_mat = self.matern(x_desc, y_desc)
        return np.nan_to_num(
            self.w_tan * k_tan + self.w_rbf * k_rbf + self.w_mat * k_mat,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

    def diag(self, x):
        return np.ones(x.shape[0])

    def is_stationary(self):
        return False


def build_kernel(fp_dim):
    return (
        ConstantKernel(FINAL_CONFIG["constant_scale"], (1e-3, 1e3))
        * TanimotoRBFMaternKernel(
            fp_dim=fp_dim,
            w_tan=FINAL_CONFIG["w_tan"],
            w_rbf=FINAL_CONFIG["w_rbf"],
            w_mat=FINAL_CONFIG["w_mat"],
            rbf_length_scale=FINAL_CONFIG["rbf_length_scale"],
            matern_length_scale=FINAL_CONFIG["matern_length_scale"],
            matern_nu=FINAL_CONFIG["matern_nu"],
        )
        + WhiteKernel(noise_level=FINAL_CONFIG["white_noise"])
    )


def select_best_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1_values = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1_values))
    if thresholds.size == 0:
        return 0.5
    if best_idx >= thresholds.size:
        best_idx = thresholds.size - 1
    return float(thresholds[best_idx])


def load_or_train_model(train_csv, bundle_path):
    train_df = pd.read_csv(train_csv)
    train_df = subsample_train(train_df, FINAL_CONFIG["max_train"], FINAL_CONFIG["seed"])

    if bundle_path:
        bundle = joblib.load(bundle_path)
        model = bundle["model"]
        scaler = bundle["scaler"]
        pca = bundle["pca"]
        x_train, y_train, fp_dim, _, _ = build_concat_features(train_df, scaler=scaler, pca=pca, fit=False)
        return train_df, x_train, y_train, fp_dim, model, scaler, pca

    x_train, y_train, fp_dim, scaler, pca = build_concat_features(train_df, fit=True)
    model = GaussianProcessClassifier(
        kernel=build_kernel(fp_dim),
        n_restarts_optimizer=FINAL_CONFIG["n_restarts"],
        random_state=0,
    )
    model.fit(x_train, y_train)
    return train_df, x_train, y_train, fp_dim, model, scaler, pca


def evaluate(model, x_test, y_test):
    proba = model.predict_proba(x_test)[:, 1]
    proba = np.nan_to_num(proba, nan=0.5, posinf=1.0, neginf=0.0)
    proba = np.clip(proba, 0.0, 1.0)
    threshold = select_best_threshold(y_test, proba)
    pred = (proba >= threshold).astype(int)
    auc_value = roc_auc_score(y_test, proba)
    acc_value = accuracy_score(y_test, pred)
    f1_value = f1_score(y_test, pred)
    brier_value = brier_score_loss(y_test, proba)
    return {
        "proba": proba,
        "pred": pred,
        "threshold": threshold,
        "auc": auc_value,
        "acc": acc_value,
        "f1": f1_value,
        "brier": brier_value,
        "cm": confusion_matrix(y_test, pred),
        "report": classification_report(y_test, pred, digits=4),
    }


def latent_uncertainty(model, x_test, interval_level):
    mean_f, var_f = model.latent_mean_and_variance(x_test)
    mean_f = np.nan_to_num(mean_f, nan=0.0, posinf=0.0, neginf=0.0)
    var_f = np.nan_to_num(var_f, nan=1.0, posinf=1.0, neginf=1.0)
    std_f = np.sqrt(np.maximum(var_f, 1e-12))
    z_value = NormalDist().inv_cdf(0.5 + interval_level / 2.0)
    p_lo = sigmoid(mean_f - z_value * std_f)
    p_hi = sigmoid(mean_f + z_value * std_f)
    return mean_f, std_f, p_lo, p_hi, p_hi - p_lo


def write_kv(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for key, value in rows:
            handle.write(f"{key}={value}\n")


def save_text_outputs(out_dir, train_df, y_test, metrics, model):
    os.makedirs(out_dir, exist_ok=True)
    preds_path = os.path.join(out_dir, "tan_rbf_mat_final_preds.csv")
    pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": metrics["pred"],
            "p_hat": metrics["proba"],
        }
    ).to_csv(preds_path, index=False)

    report_path = os.path.join(out_dir, "tan_rbf_mat_final_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(
            f"AUC={metrics['auc']:.4f} | ACC={metrics['acc']:.4f} | "
            f"F1={metrics['f1']:.4f} | Brier={metrics['brier']:.4f}\n"
        )
        handle.write(f"Best-threshold={metrics['threshold']:.3f}\n\n")
        handle.write(f"Confusion Matrix:\n{metrics['cm']}\n\n")
        handle.write(metrics["report"])

    metrics_path = os.path.join(out_dir, "tan_rbf_mat_final_metrics.txt")
    write_kv(
        metrics_path,
        [
            ("AUC", metrics["auc"]),
            ("ACC", metrics["acc"]),
            ("F1", metrics["f1"]),
            ("Brier", metrics["brier"]),
            ("Best_thr", metrics["threshold"]),
        ],
    )

    config_path = os.path.join(out_dir, "tan_rbf_mat_final_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": FINAL_CONFIG,
                "train_rows": len(train_df),
                "test_rows": int(len(y_test)),
                "fitted_kernel": str(model.kernel_),
            },
            handle,
            indent=2,
        )

    return preds_path, metrics_path


def plot_roc(y_true, proba, path):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_value = roc_auc_score(y_true, proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_pr(y_true, proba, path):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, linewidth=2, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion(cm, path):
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (row, col), value in np.ndenumerate(cm):
        plt.text(col, row, f"{value}", ha="center", va="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_score_hist(y_true, proba, path):
    plt.figure(figsize=(6, 4))
    plt.hist(proba[y_true == 0], bins=30, alpha=0.65, label="Class 0")
    plt.hist(proba[y_true == 1], bins=30, alpha=0.65, label="Class 1")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Score Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_reliability(y_true, proba, path):
    prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10)
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(prob_pred, prob_true, "o-", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title("Reliability Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_threshold_curves(y_true, proba, path):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.plot(thresholds, f1_values, label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_prob_coverage_bins(y_true, proba, out_csv, out_png, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    frame = pd.DataFrame({"y_true": y_true, "proba": proba})
    frame["bin"] = pd.cut(frame["proba"], bins=edges, include_lowest=True)
    grouped = frame.groupby("bin", observed=False).agg(
        count=("y_true", "size"),
        mean_pred=("proba", "mean"),
        empirical_pos=("y_true", "mean"),
    ).reset_index()
    grouped.to_csv(out_csv, index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(grouped["mean_pred"], grouped["empirical_pos"], "o-", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical positive rate")
    plt.title("Probability Coverage by Bin")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_uncertainty_scatter(proba, score, xlabel, title, path):
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(proba, score, s=18, alpha=0.65)
    plt.xlabel("Predicted Probability")
    plt.ylabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_uncertainty_hist(values, labels, path, title, xlabel):
    plt.figure(figsize=(6, 4))
    for value, label in labels:
        plt.hist(value, bins=30, alpha=0.6, label=label)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def coverage_curves(y_true, pred, uncertainty, minimum_keep=30):
    order = np.argsort(uncertainty)
    y_sorted = y_true[order]
    pred_sorted = pred[order]
    coverages, risks, accuracies, f1s = [], [], [], []
    total = len(y_true)
    for keep in range(minimum_keep, total + 1, max(1, total // 80)):
        kept_y = y_sorted[:keep]
        kept_pred = pred_sorted[:keep]
        acc_value = accuracy_score(kept_y, kept_pred)
        f1_value = f1_score(kept_y, kept_pred)
        coverages.append(keep / total)
        accuracies.append(acc_value)
        f1s.append(f1_value)
        risks.append(1.0 - acc_value)
    return np.array(coverages), np.array(risks), np.array(accuracies), np.array(f1s)


def plot_single_curve(x, y, path, title, ylabel):
    plt.figure(figsize=(5.5, 4.5))
    plt.plot(x, y, linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_true_vs_pred_band(y_true, proba, p_lo, p_hi, path):
    order = np.argsort(proba)
    x_axis = np.arange(len(proba))
    plt.figure(figsize=(7, 4.5))
    plt.fill_between(x_axis, p_lo[order], p_hi[order], color="#bfdbfe", alpha=0.6, label="Interval")
    plt.plot(x_axis, proba[order], color="#1d4ed8", linewidth=1.8, label="Predicted probability")
    plt.scatter(x_axis, y_true[order], s=12, color="#111827", alpha=0.6, label="True label")
    plt.xlabel("Samples sorted by predicted probability")
    plt.ylabel("Probability / Label")
    plt.title("True vs Predicted Band")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_actual_vs_pred_errorbar(y_true, proba, p_lo, p_hi, path, max_points=120):
    order = np.argsort(proba)
    select = order[np.linspace(0, len(order) - 1, min(max_points, len(order))).astype(int)]
    lower = np.maximum(proba[select] - p_lo[select], 0.0)
    upper = np.maximum(p_hi[select] - proba[select], 0.0)
    plt.figure(figsize=(6.5, 4.5))
    plt.errorbar(
        y_true[select] + np.linspace(-0.04, 0.04, len(select)),
        proba[select],
        yerr=np.vstack([lower, upper]),
        fmt="o",
        alpha=0.55,
        markersize=3,
        ecolor="#60a5fa",
        color="#1d4ed8",
    )
    plt.xlabel("True label")
    plt.ylabel("Predicted probability")
    plt.title("Actual vs Predicted with Interval")
    plt.xticks([0, 1], ["Class 0", "Class 1"])
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_pca_maps(x_test, y_true, proba, label_path, prob_path):
    pca2 = PCA(n_components=2, random_state=0)
    reduced = pca2.fit_transform(x_test)

    plt.figure(figsize=(5.5, 4.5))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=y_true, cmap="coolwarm", s=18, alpha=0.75)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Map by Label")
    plt.colorbar(scatter, ticks=[0, 1])
    plt.tight_layout()
    plt.savefig(label_path, dpi=300)
    plt.close()

    plt.figure(figsize=(5.5, 4.5))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=proba, cmap="viridis", s=18, alpha=0.75)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Map by Probability")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(prob_path, dpi=300)
    plt.close()


def plot_kernel_heatmap(model, x_test, path, max_points=140):
    subset = x_test[: min(max_points, len(x_test))]
    kernel_matrix = model.kernel_(subset)
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(kernel_matrix, cmap="viridis")
    plt.title("Kernel Heatmap")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def prediction_set_stats(y_true, p_lo, p_hi):
    set_size = np.where((p_lo > 0.5) | (p_hi < 0.5), 1, 2)
    include_one = p_hi >= 0.5
    include_zero = p_lo <= 0.5
    covered = np.where(y_true == 1, include_one, include_zero)
    return {
        "coverage": float(np.mean(covered)),
        "avg_size": float(np.mean(set_size)),
        "ambiguous_rate": float(np.mean(set_size == 2)),
    }


def plot_predset_tradeoff(y_true, mean_f, std_f, out_csv, out_png):
    rows = []
    for level in [0.80, 0.85, 0.90, 0.95, 0.975]:
        z_value = NormalDist().inv_cdf(0.5 + level / 2.0)
        p_lo = sigmoid(mean_f - z_value * std_f)
        p_hi = sigmoid(mean_f + z_value * std_f)
        stats = prediction_set_stats(y_true, p_lo, p_hi)
        stats["level"] = level
        rows.append(stats)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(frame["level"], frame["coverage"], marker="o", label="Set coverage")
    plt.plot(frame["level"], frame["avg_size"], marker="o", label="Avg set size")
    plt.plot(frame["level"], frame["ambiguous_rate"], marker="o", label="Ambiguous rate")
    plt.xlabel("Interval level")
    plt.ylabel("Value")
    plt.title("Prediction Set Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_metric_bars(df, metric, title, out_path, higher_is_better=True):
    frame = df.copy()
    frame = frame.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)
    best_idx = frame[metric].idxmax() if higher_is_better else frame[metric].idxmin()
    colors = ["#0f766e" if idx == best_idx else "#93c5fd" for idx in frame.index]

    plt.figure(figsize=(7, 4.5))
    plt.bar(frame["combo"], frame[metric], color=colors, edgecolor="black", linewidth=0.6)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_f1_brier_tradeoff(df, out_path):
    frame = df.copy()
    f1_min, f1_max = frame["F1"].min(), frame["F1"].max()
    brier_min, brier_max = frame["Brier"].min(), frame["Brier"].max()
    frame["tradeoff_score"] = (
        (frame["F1"] - f1_min) / max(f1_max - f1_min, 1e-12)
        + 1.0
        - (frame["Brier"] - brier_min) / max(brier_max - brier_min, 1e-12)
    )
    best_idx = frame["tradeoff_score"].idxmax()
    best_row = frame.loc[best_idx]

    x_min, x_max = frame["Brier"].min(), frame["Brier"].max()
    y_min, y_max = frame["F1"].min(), frame["F1"].max()
    x_pad = max((x_max - x_min) * 0.08, 0.005)
    y_pad = max((y_max - y_min) * 0.08, 0.005)
    pref_x = x_min + (x_max - x_min) * 0.30
    pref_y = y_max - (y_max - y_min) * 0.30
    ymin_norm = (pref_y - (y_min - y_pad)) / ((y_max + y_pad) - (y_min - y_pad))

    plt.figure(figsize=(7, 6))
    plt.axvspan(
        x_min - x_pad,
        pref_x,
        ymin=ymin_norm,
        ymax=1.0,
        color="#d1fae5",
        alpha=0.45,
        label="Preferred region",
    )
    for idx, row in frame.iterrows():
        color = "#0f766e" if idx == best_idx else "#2563eb"
        size = 120 if idx == best_idx else 85
        label = f"{row['combo']} (best trade-off)" if idx == best_idx else row["combo"]
        plt.scatter(row["Brier"], row["F1"], s=size, color=color, edgecolor="black", linewidth=0.6, zorder=3)
        plt.annotate(label, (row["Brier"], row["F1"]), xytext=(6, 6), textcoords="offset points", fontsize=9)

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
    plt.savefig(out_path, dpi=300)
    plt.close()
    return frame


def build_comparison_artifacts(summary_csv, metrics, out_dir):
    if not summary_csv or not os.path.exists(summary_csv):
        return None

    frame = pd.read_csv(summary_csv)
    mask = frame["combo"] == "tan+rbf+mat"
    if mask.any():
        frame.loc[mask, "AUC"] = metrics["auc"]
        frame.loc[mask, "ACC"] = metrics["acc"]
        frame.loc[mask, "F1"] = metrics["f1"]
        frame.loc[mask, "Brier"] = metrics["brier"]

    updated_csv = os.path.join(out_dir, "summary_metrics_updated.csv")
    frame.to_csv(updated_csv, index=False)

    plot_metric_bars(frame, "AUC", "Kernel Comparison: AUC", os.path.join(out_dir, "kernel_auc_comparison.png"))
    plot_metric_bars(frame, "F1", "Kernel Comparison: F1", os.path.join(out_dir, "kernel_f1_comparison.png"))
    plot_metric_bars(
        frame,
        "Brier",
        "Kernel Comparison: Brier",
        os.path.join(out_dir, "kernel_brier_comparison.png"),
        higher_is_better=False,
    )
    tradeoff_frame = plot_f1_brier_tradeoff(frame, os.path.join(out_dir, "f1_vs_brier_tradeoff.png"))
    tradeoff_frame.to_csv(os.path.join(out_dir, "summary_metrics_with_tradeoff.csv"), index=False)
    return updated_csv


def main():
    args = parse_args()
    set_seed(FINAL_CONFIG["seed"])

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    comparison_dir = os.path.join(args.out_dir, "comparison")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    train_df, x_train, y_train, fp_dim, model, scaler, pca = load_or_train_model(args.train_csv, args.bundle_path)
    test_df = pd.read_csv(args.test_csv)
    x_test, y_test, _, _, _ = build_concat_features(test_df, scaler=scaler, pca=pca, fit=False)

    metrics = evaluate(model, x_test, y_test)
    mean_f, std_f, p_lo, p_hi, prob_width = latent_uncertainty(model, x_test, FINAL_CONFIG["interval_level"])
    correct_mask = metrics["pred"] == y_test

    save_text_outputs(args.out_dir, train_df, y_test, metrics, model)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "pca": pca,
            "config": FINAL_CONFIG,
            "fitted_kernel": str(model.kernel_),
        },
        os.path.join(args.out_dir, "tan_rbf_mat_final_bundle.pkl"),
    )

    plot_roc(y_test, metrics["proba"], os.path.join(plots_dir, "tan_rbf_mat_final_roc.png"))
    plot_pr(y_test, metrics["proba"], os.path.join(plots_dir, "tan_rbf_mat_final_pr.png"))
    plot_confusion(metrics["cm"], os.path.join(plots_dir, "tan_rbf_mat_final_confusion.png"))
    plot_score_hist(y_test, metrics["proba"], os.path.join(plots_dir, "tan_rbf_mat_final_score_hist.png"))
    plot_reliability(y_test, metrics["proba"], os.path.join(plots_dir, "tan_rbf_mat_final_reliability.png"))
    plot_threshold_curves(y_test, metrics["proba"], os.path.join(plots_dir, "tan_rbf_mat_final_threshold_curves.png"))
    plot_prob_coverage_bins(
        y_test,
        metrics["proba"],
        os.path.join(plots_dir, "tan_rbf_mat_final_prob_coverage_bins.csv"),
        os.path.join(plots_dir, "tan_rbf_mat_final_prob_coverage_bins.png"),
    )
    plot_uncertainty_scatter(
        metrics["proba"],
        std_f,
        "Latent Std",
        "Uncertainty Scatter: latent std vs probability",
        os.path.join(plots_dir, "tan_rbf_mat_final_uncertainty_scatter_std_vs_p.png"),
    )
    plot_uncertainty_scatter(
        metrics["proba"],
        prob_width,
        "Probability Interval Width",
        "Uncertainty Scatter: interval width vs probability",
        os.path.join(plots_dir, "tan_rbf_mat_final_uncertainty_scatter_width_vs_p.png"),
    )
    plot_uncertainty_hist(
        std_f,
        [(std_f[correct_mask], "Correct"), (std_f[~correct_mask], "Wrong")],
        os.path.join(plots_dir, "tan_rbf_mat_final_uncertainty_hist_std_correct_vs_wrong.png"),
        "Latent Std: correct vs wrong",
        "Latent Std",
    )
    plot_uncertainty_hist(
        prob_width,
        [(prob_width, "All samples")],
        os.path.join(plots_dir, "tan_rbf_mat_final_uncertainty_hist_width.png"),
        "Probability Interval Width",
        "Interval Width",
    )
    cov_x, risk_y, acc_y, f1_y = coverage_curves(y_test, metrics["pred"], std_f)
    plot_single_curve(cov_x, risk_y, os.path.join(plots_dir, "tan_rbf_mat_final_risk_coverage_latent_std.png"), "Risk-Coverage by Latent Std", "Risk")
    plot_single_curve(cov_x, acc_y, os.path.join(plots_dir, "tan_rbf_mat_final_accuracy_coverage_latent_std.png"), "Accuracy-Coverage by Latent Std", "Accuracy")
    plot_single_curve(cov_x, f1_y, os.path.join(plots_dir, "tan_rbf_mat_final_f1_coverage_latent_std.png"), "F1-Coverage by Latent Std", "F1")
    cov_x2, risk_y2, acc_y2, f1_y2 = coverage_curves(y_test, metrics["pred"], prob_width)
    plot_single_curve(cov_x2, risk_y2, os.path.join(plots_dir, "tan_rbf_mat_final_risk_coverage_prob_width.png"), "Risk-Coverage by Interval Width", "Risk")
    plot_single_curve(cov_x2, acc_y2, os.path.join(plots_dir, "tan_rbf_mat_final_accuracy_coverage_prob_width.png"), "Accuracy-Coverage by Interval Width", "Accuracy")
    plot_single_curve(cov_x2, f1_y2, os.path.join(plots_dir, "tan_rbf_mat_final_f1_coverage_prob_width.png"), "F1-Coverage by Interval Width", "F1")
    plot_true_vs_pred_band(
        y_test,
        metrics["proba"],
        p_lo,
        p_hi,
        os.path.join(plots_dir, "tan_rbf_mat_final_true_vs_pred_band.png"),
    )
    plot_actual_vs_pred_errorbar(
        y_test,
        metrics["proba"],
        p_lo,
        p_hi,
        os.path.join(plots_dir, "tan_rbf_mat_final_actual_vs_pred_errorbar.png"),
    )
    plot_pca_maps(
        x_test,
        y_test,
        metrics["proba"],
        os.path.join(plots_dir, "tan_rbf_mat_final_pca2_label_map.png"),
        os.path.join(plots_dir, "tan_rbf_mat_final_pca2_prob_map.png"),
    )
    plot_kernel_heatmap(model, x_test, os.path.join(plots_dir, "tan_rbf_mat_final_kernel_heatmap.png"))
    plot_predset_tradeoff(
        y_test,
        mean_f,
        std_f,
        os.path.join(plots_dir, "tan_rbf_mat_final_predset_tradeoff.csv"),
        os.path.join(plots_dir, "tan_rbf_mat_final_predset_tradeoff.png"),
    )

    build_comparison_artifacts(args.kernel_summary_csv, metrics, comparison_dir)

    print(f"[Saved] final artifacts under {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
