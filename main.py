#!/usr/bin/env python3
"""
Run Bayesian network pruning experiments: alarm only, synthetic only, or both.
Pruning methods: Wavelet, BIC, AIC, BDs, CSI.
At the end shows the comparison table and progress plots.
"""

import argparse
import sys
import os

# Allow running as script from repo root or from package dir
if __name__ == "__main__" and __package__ is None:
    _dir = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_dir)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    __package__ = "bayesian_pruning_project"

try:
    from . import config as config_module
    from .data_alarm import load_alarm_data
    from .data_synthetic import load_synthetic_from_config
    from .pruning_wavelet import pruning_l2_wavelet
    from .pruning_score import score_pruning
    from .pruning_structural import structural_error_pruning
    from .comparison import run_and_print_comparison, plot_pruning_progress
except ImportError:
    from bayesian_pruning_project import config as config_module
    from bayesian_pruning_project.data_alarm import load_alarm_data
    from bayesian_pruning_project.data_synthetic import load_synthetic_from_config
    from bayesian_pruning_project.pruning_wavelet import pruning_l2_wavelet
    from bayesian_pruning_project.pruning_score import score_pruning
    from bayesian_pruning_project.pruning_structural import structural_error_pruning
    from bayesian_pruning_project.comparison import run_and_print_comparison, plot_pruning_progress


def run_alarm():
    print("\n" + "=" * 60 + "\n  ALARM MODEL\n" + "=" * 60)
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
    print("\n" + "=" * 60 + "\n  SYNTHETIC MODEL\n" + "=" * 60)
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
    args = parser.parse_args()

    if args.both:
        args.alarm = True
        args.synthetic = True
    if not args.alarm and not args.synthetic:
        parser.error("Specify at least one of: --alarm, --synthetic, --both")

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
