#!/usr/bin/env python3
"""
Run Bayesian network pruning experiments: alarm only, synthetic only, or both.
Pruning methods: Wavelet, BIC, AIC, BDs, CSI.
At the end shows the comparison table and progress plots.
"""

import argparse
import sys
import os
import importlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

# When run as a script (e.g. python main.py from the project folder), the "package"
# is the folder that contains main.py — e.g. CurrentTrendsinIndustrialMathProject on Colab
# or bayesian_pruning_project locally. We add the parent of that folder to sys.path
# and set __package__ to the folder name so imports work no matter what the folder is called.
if __name__ == "__main__" and __package__ is None:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_script_dir)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    __package__ = os.path.basename(_script_dir)

# Suppress pgmpy INFO/WARNING (e.g. "Probability values don't sum to 1", "Datatype inferred")
import logging
logging.getLogger("pgmpy").setLevel(logging.ERROR)

try:
    from . import config as config_module
    from .data_alarm import load_alarm_data
    from .data_synthetic import load_synthetic_from_config
    from .pruning_wavelet import pruning_l2_wavelet
    from .pruning_score import score_pruning
    from .pruning_structural import structural_error_pruning
    from .comparison import (
        run_and_print_comparison,
        plot_pruning_progress,
        build_comparison_from_last_step,
        print_comparison,
    )
    from .pipeline import run_alarm_worker, run_synthetic_worker
except ImportError:
    # Fallback when relative imports fail: import by package name (the folder name).
    _pkg = __package__ or os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    config_module = importlib.import_module(_pkg + ".config")
    load_alarm_data = getattr(importlib.import_module(_pkg + ".data_alarm"), "load_alarm_data")
    load_synthetic_from_config = getattr(importlib.import_module(_pkg + ".data_synthetic"), "load_synthetic_from_config")
    pruning_l2_wavelet = getattr(importlib.import_module(_pkg + ".pruning_wavelet"), "pruning_l2_wavelet")
    score_pruning = getattr(importlib.import_module(_pkg + ".pruning_score"), "score_pruning")
    structural_error_pruning = getattr(importlib.import_module(_pkg + ".pruning_structural"), "structural_error_pruning")
    run_and_print_comparison = getattr(importlib.import_module(_pkg + ".comparison"), "run_and_print_comparison")
    plot_pruning_progress = getattr(importlib.import_module(_pkg + ".comparison"), "plot_pruning_progress")
    build_comparison_from_last_step = getattr(importlib.import_module(_pkg + ".comparison"), "build_comparison_from_last_step")
    print_comparison = getattr(importlib.import_module(_pkg + ".comparison"), "print_comparison")
    run_alarm_worker = getattr(importlib.import_module(_pkg + ".pipeline"), "run_alarm_worker")
    run_synthetic_worker = getattr(importlib.import_module(_pkg + ".pipeline"), "run_synthetic_worker")


def _aggregate_histories(list_of_histories_dicts):
    """Average metrics across runs per (method, step). Steps aligned by step number."""
    method_order = ("Wavelet", "BIC", "AIC", "BDs", "CSI")
    metric_cols = ["num_edges", "ll_score", "kl_score", "structure_score", "pred_accuracy", "collider_recall", "global_ace_diff", "interventional_kl_mean", "score"]
    aggregated = {}
    for method in method_order:
        all_rows = []
        for run_histories in list_of_histories_dicts:
            if method not in run_histories:
                continue
            for rec in run_histories[method]:
                step = rec.get("step")
                if step is None:
                    continue
                row = {"step": step}
                for k in metric_cols:
                    if k in rec and rec[k] is not None:
                        try:
                            row[k] = float(rec[k])
                        except (TypeError, ValueError):
                            pass
                all_rows.append(row)
        if not all_rows:
            continue
        df_run = pd.DataFrame(all_rows)
        by_step = df_run.groupby("step", as_index=False).mean()
        aggregated[method] = by_step.to_dict("records")
    return aggregated


def _aggregate_comparison_dfs(list_of_dfs):
    """Mean and std per method per metric across dataframes. Returns (df_mean, df_std)."""
    if not list_of_dfs:
        return pd.DataFrame(), pd.DataFrame()
    concat = pd.concat(list_of_dfs, ignore_index=True)
    method_col = "method"
    numeric_cols = [c for c in concat.columns if c != method_col and pd.api.types.is_numeric_dtype(concat[c])]
    if not numeric_cols:
        return concat.groupby(method_col, as_index=False).first(), pd.DataFrame()
    mean_df = concat.groupby(method_col, as_index=False)[numeric_cols].mean()
    std_df = concat.groupby(method_col, as_index=False)[numeric_cols].std()
    return mean_df, std_df


def run_alarm():
    true_model, pruned_model, train_data, eval_data, target_var, interventions = load_alarm_data()

    wavelet_model, wavelet_history = pruning_l2_wavelet(
        true_model=true_model,
        pruned_model=pruned_model,
        data=train_data,
        evaluate_data=eval_data,
        target_var=target_var,
        interventions=interventions,
    )
    BIC_model, BIC_history = score_pruning(
        true_model=true_model,
        pruned_model=pruned_model,
        data=train_data,
        evaluate_data=eval_data,
        score_fn="bic-d",
        target_var=target_var,
        interventions=interventions,
    )
    AIC_model, AIC_history = score_pruning(
        true_model=true_model,
        pruned_model=pruned_model,
        data=train_data,
        evaluate_data=eval_data,
        score_fn="aic-d",
        target_var=target_var,
        interventions=interventions,
    )
    BDs_model, BDs_history = score_pruning(
        true_model=true_model,
        pruned_model=pruned_model,
        data=train_data,
        evaluate_data=eval_data,
        score_fn="bds",
        target_var=target_var,
        interventions=interventions,
    )
    csi_model, csi_history = structural_error_pruning(
        true_model=true_model,
        pruned_model=pruned_model,
        data=train_data,
        evaluate_data=eval_data,
        target_var=target_var,
        interventions=interventions,
    )

    df = run_and_print_comparison(
        wavelet_history=wavelet_history,
        BIC_history=BIC_history,
        AIC_history=AIC_history,
        BDs_history=BDs_history,
        csi_history=csi_history,
    )
    fig, axes = plot_pruning_progress({
        "Wavelet": wavelet_history,
        "BIC": BIC_history,
        "AIC": AIC_history,
        "BDs": BDs_history,
        "CSI": csi_history,
    })
    if fig is not None:
        fig.suptitle("ALARM — Pruning progress", fontsize=12)
        fig.savefig("alarm_pruning_progress.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to alarm_pruning_progress.png")
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass
    return df


def run_synthetic(config_path=None):
    true_model, pruned_model, train_data, eval_data, target_var, interventions = load_synthetic_from_config(config_path)

    wavelet_model, wavelet_history = pruning_l2_wavelet(
        true_model=true_model,
        pruned_model=pruned_model,
        data=train_data,
        evaluate_data=eval_data,
        target_var=target_var,
        interventions=interventions,
    )
    BIC_model, BIC_history = score_pruning(
        true_model=true_model,
        pruned_model=pruned_model,
        data=train_data,
        evaluate_data=eval_data,
        score_fn="bic-d",
        target_var=target_var,
        interventions=interventions,
    )
    AIC_model, AIC_history = score_pruning(
        true_model=true_model,
        pruned_model=pruned_model,
        data=train_data,
        evaluate_data=eval_data,
        score_fn="aic-d",
        target_var=target_var,
        interventions=interventions,
    )
    BDs_model, BDs_history = score_pruning(
        true_model=true_model,
        pruned_model=pruned_model,
        data=train_data,
        evaluate_data=eval_data,
        score_fn="bds",
        target_var=target_var,
        interventions=interventions,
    )
    csi_model, csi_history = structural_error_pruning(
        true_model=true_model,
        pruned_model=pruned_model,
        data=train_data,
        evaluate_data=eval_data,
        target_var=target_var,
        interventions=interventions,
    )

    df = run_and_print_comparison(
        wavelet_history=wavelet_history,
        BIC_history=BIC_history,
        AIC_history=AIC_history,
        BDs_history=BDs_history,
        csi_history=csi_history,
    )
    fig, axes = plot_pruning_progress({
        "Wavelet": wavelet_history,
        "BIC": BIC_history,
        "AIC": AIC_history,
        "BDs": BDs_history,
        "CSI": csi_history,
    })
    if fig is not None:
        fig.suptitle("Synthetic — Pruning progress", fontsize=12)
        fig.savefig("synthetic_pruning_progress.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to synthetic_pruning_progress.png")
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass
    return df


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian network pruning: alarm, synthetic, or both.")
    parser.add_argument("--alarm", action="store_true", help="Run experiments on ALARM model only")
    parser.add_argument("--synthetic", action="store_true", help="Run experiments on synthetic model only")
    parser.add_argument("--both", action="store_true", help="Run both ALARM and synthetic")
    parser.add_argument("--config", default=None, help="Path to synthetic bayesian_config.json (default: config.CONFIG_PATH)")
    parser.add_argument("--runs", type=int, default=1, metavar="N", help="Run pipeline N times and report averaged results (default: 1)")
    args = parser.parse_args()

    if args.both:
        args.alarm = True
        args.synthetic = True
    if not args.alarm and not args.synthetic:
        parser.error("Specify at least one of: --alarm, --synthetic, --both")

    base_seed = getattr(config_module, "RANDOM_STATE", 42)

    if args.runs > 1:
        n = args.runs
        n_workers = min(n, os.cpu_count() or 4)
        if args.alarm:
            print("Running ALARM pipeline {} times in parallel ({} workers, seeds {}..{})...".format(
                n, n_workers, base_seed, base_seed + n - 1))
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(run_alarm_worker, [base_seed + i for i in range(n)]))
            all_histories = [r[0] for r in results]
            all_dfs = [r[1] for r in results]
            mean_df, std_df = _aggregate_comparison_dfs(all_dfs)
            print("\n--- Averaged comparison (mean over {} runs) ---".format(n))
            print_comparison(mean_df, mark_best=True)
            if not std_df.empty:
                print("\n--- Std over {} runs ---".format(n))
                print(std_df.round(4).to_string(index=False))
            agg_hist = _aggregate_histories(all_histories)
            fig, _ = plot_pruning_progress(agg_hist)
            if fig is not None:
                fig.suptitle("ALARM — Pruning progress (mean over {} runs)".format(n), fontsize=12)
                fig.savefig("alarm_pruning_progress.png", dpi=150, bbox_inches="tight")
                print("\nPlot saved to alarm_pruning_progress.png")
                try:
                    import matplotlib.pyplot as plt
                    plt.show()
                except Exception:
                    pass
        if args.synthetic:
            config_path = args.config or getattr(config_module, "CONFIG_PATH", "bayesian_config.json")
            if not os.path.isfile(config_path):
                print("[WARN] Synthetic config not found: {}".format(config_path))
            else:
                print("Running synthetic pipeline {} times in parallel ({} workers, seeds {}..{})...".format(
                    n, n_workers, base_seed, base_seed + n - 1))
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(
                        run_synthetic_worker,
                        [(config_path, base_seed + i) for i in range(n)],
                    ))
                all_histories = [r[0] for r in results]
                all_dfs = [r[1] for r in results]
                mean_df, std_df = _aggregate_comparison_dfs(all_dfs)
                print("\n--- Averaged comparison (mean over {} runs) ---".format(n))
                print_comparison(mean_df, mark_best=True)
                if not std_df.empty:
                    print("\n--- Std over {} runs ---".format(n))
                    print(std_df.round(4).to_string(index=False))
                agg_hist = _aggregate_histories(all_histories)
                fig, _ = plot_pruning_progress(agg_hist)
                if fig is not None:
                    fig.suptitle("Synthetic — Pruning progress (mean over {} runs)".format(n), fontsize=12)
                    fig.savefig("synthetic_pruning_progress.png", dpi=150, bbox_inches="tight")
                    print("\nPlot saved to synthetic_pruning_progress.png")
                    try:
                        import matplotlib.pyplot as plt
                        plt.show()
                    except Exception:
                        pass
        return

    if args.alarm:
        run_alarm()
    if args.synthetic:
        config_path = args.config or getattr(config_module, "CONFIG_PATH", "bayesian_config.json")
        if not os.path.isfile(config_path):
            print(f"[WARN] Synthetic config not found: {config_path}. Copy bayesian_config.json to project root or set CONFIG_PATH.")
        else:
            run_synthetic(config_path=config_path)


if __name__ == "__main__":
    main()
