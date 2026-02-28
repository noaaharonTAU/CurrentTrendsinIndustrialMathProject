"""
Compare pruning methods (Wavelet, BIC, AIC, BDs, CSI) and plot progress.
Builds comparison table from last step of each history and plots metrics vs step.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_METHOD_ORDER = ("Wavelet", "BIC", "AIC", "BDs", "CSI") = ("Wavelet", "BIC", "AIC", "BDs", "CSI")

PROGRESS_METRICS = (
    ("num_edges", "Number of edges", "lower"),
    ("ll_score", "Negative Log-likelihood", "higher"),
    ("kl_score", "KL divergence (to true)", "lower"),
    ("structure_score", "Structural error (CI disagreement)", "lower"),
    ("score", "Each method score", "higher"),
    ("pred_accuracy", "Prediction accuracy", "higher"),
    ("global_ace_diff", "Global ACE difference", "lower"),
    ("collider_recall", "Collider recall", "higher"),
    ("interventional_kl_mean", "Interventional KL (mean)", "lower"),
)


def build_comparison_from_last_step(histories: dict) -> pd.DataFrame:
    """Build comparison table from the LAST stored step of each pruning method history."""
    rows = []
    for method_name, history in histories.items():
        if not isinstance(history, list) or len(history) == 0:
            raise ValueError(f"{method_name} history is empty or invalid")
        last = history[-1]
        row = {
            "method": method_name,
            "step": last.get("step"),
            "num_edges": last.get("num_edges"),
            "neg_ll": last.get("ll_score"),
            "kl_div": last.get("kl_score"),
            "pred_accuracy": last.get("pred_accuracy"),
            "structural_error": last.get("structure_score"),
            "collider_recall": last.get("collider_recall"),
            "interventional_kl_mean": last.get("interventional_kl_mean"),
            "global_ace_diff": last.get("global_ace_diff"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def print_comparison(
    df: pd.DataFrame,
    method_col: str = "method",
    predictive_cols: tuple = ("neg_ll", "kl_div", "pred_accuracy"),
    causal_cols: tuple = ("structural_error", "collider_recall", "interventional_kl_mean", "global_ace_diff"),
    lower_better: tuple = ("neg_ll", "kl_div", "structural_error", "interventional_kl_mean", "global_ace_diff"),
    round_digits: int = 4,
    mark_best: bool = True,
) -> None:
    """Print comparison table with Predictive / Causal sections and best markers (*)."""
    if df is None or len(df) == 0:
        print("No comparison data.")
        return
    df = df.copy()
    if method_col not in df.columns and len(df.columns) > 0:
        df.insert(0, method_col, df.index.astype(str))
    for c in list(predictive_cols) + list(causal_cols):
        if c in df.columns and np.issubdtype(df[c].dtype, np.floating):
            df[c] = df[c].round(round_digits)
    if mark_best:
        for c in list(predictive_cols) + list(causal_cols):
            if c not in df.columns:
                continue
            valid = pd.to_numeric(df[c], errors="coerce")
            if valid.isna().all():
                continue
            best_val = valid.min() if c in lower_better else valid.max()
            df[c] = df[c].astype(object)
            mask = (pd.to_numeric(df[c], errors="coerce") == best_val) & valid.notna()
            df.loc[mask, c] = df.loc[mask, c].astype(str) + " *"
    print("=" * 80)
    print("PRUNING METHODS COMPARISON (Wavelet, BIC, AIC, BDs, CSI)")
    print("=" * 80)
    pred = [c for c in predictive_cols if c in df.columns]
    if pred:
        print("\n--- Predictive quality ---")
        print(df[[method_col] + pred].to_string(index=False))
    caus = [c for c in causal_cols if c in df.columns]
    if caus:
        print("\n--- Causal quality ---")
        print(df[[method_col] + caus].to_string(index=False))
    print("\n" + "-" * 80)
    if mark_best:
        print("* = best in column")
    print("=" * 80)


def run_and_print_comparison(
    wavelet_history: list = None,
    BIC_history: list = None,
    AIC_history: list = None,
    BDs_history: list = None,
    csi_history: list = None,
    method_order: tuple = None,
) -> pd.DataFrame:
    """Build comparison table from LAST STEP of stored histories and print it."""
    method_order = method_order or DEFAULT_METHOD_ORDER
    histories = {}
    if wavelet_history is not None:
        histories["Wavelet"] = wavelet_history
    if BIC_history is not None:
        histories["BIC"] = BIC_history
    if AIC_history is not None:
        histories["AIC"] = AIC_history
    if BDs_history is not None:
        histories["BDs"] = BDs_history
    if csi_history is not None:
        histories["CSI"] = csi_history

    if not histories:
        print("No histories provided.")
        return pd.DataFrame()

    ordered = [(k, histories[k]) for k in method_order if k in histories]
    histories = dict(ordered)
    df = build_comparison_from_last_step(histories)
    print_comparison(df)
    return df


def histories_to_dataframe(histories_dict):
    """Convert {method_name: history_list} to a long DataFrame."""
    rows = []
    for method_name, history in histories_dict.items():
        if not history:
            continue
        for rec in history:
            row = {"method": method_name}
            step = rec.get("step")
            if step is None:
                continue
            row["step"] = step
            for key in ("num_edges", "ll_score", "kl_score", "structure_score",
                        "pred_accuracy", "collider_recall", "global_ace_diff",
                        "interventional_kl_mean", "score"):
                if key in rec:
                    val = rec[key]
                    if isinstance(val, (list, tuple)) and len(val) == 0:
                        continue
                    row[key] = val
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def plot_pruning_progress(
    histories_dict,
    metrics=None,
    method_order=None,
    figsize_per_plot=(6, 3.5),
    ncols=2,
    sharex=True,
):
    """Plot metrics vs pruning step for each method."""
    df = histories_to_dataframe(histories_dict)
    if df.empty:
        print("No history data to plot.")
        return None, None

    method_order = method_order or DEFAULT_METHOD_ORDER
    methods = [m for m in method_order if m in df["method"].values]
    if not methods:
        methods = list(df["method"].unique())

    metric_info = {m[0]: (m[1], m[2]) for m in PROGRESS_METRICS}
    if metrics is None:
        metrics = [m[0] for m in PROGRESS_METRICS if m[0] in df.columns]
    else:
        metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        print("No plottable metrics found in history. Available:", list(df.columns))
        return None, None

    nplots = len(metrics)
    nrows = (nplots + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        sharex=sharex,
        squeeze=False,
    )
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 10)))
    method_color = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        title, _ = metric_info.get(metric, (metric.replace("_", " ").title(), ""))

        for method in methods:
            if metric == "score" and method not in ("BIC", "AIC", "BDs"):
                continue
            sub = df[df["method"] == method].sort_values("step")
            if sub.empty or metric not in sub.columns:
                continue
            y = pd.to_numeric(sub[metric], errors="coerce")
            if y.isna().all():
                continue
            ax.plot(
                sub["step"], y,
                marker="o", markersize=4,
                label=method,
                color=method_color.get(method, None),
            )
            if metric == "score":
                max_idx = y.idxmax()
                max_row = sub.loc[max_idx]
                ax.scatter(
                    max_row["step"], max_row[metric],
                    s=120, marker="*", edgecolor="black", linewidth=1.2, zorder=5,
                    color=method_color.get(method, None),
                )
        ax.set_title(title)
        ax.set_xlabel("Pruning step")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig, axes
