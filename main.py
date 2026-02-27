"""
Current Trends in Industrial Math Project — main entry point.

Run ALARM and synthetic experiments, then print comparison tables and save progress plots.

Usage (from project root):
  pip install -r requirements.txt
  python main.py

Requires: bayesian_config.json in the same directory (for synthetic model). If missing, only ALARM runs.
"""

from __future__ import print_function

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.utils import get_example_model
from sklearn.model_selection import train_test_split

# Project root = directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import bayesian_evaluation as ev
from bayesian_wavelet_pruning_fixed import pruning_l2_wavelet, _refit_model
from bayesian_score_pruning import score_pruning
from bayesian_csi_pruning import structural_error_pruning
from compare_pruning_methods import (
    run_and_print_comparison,
    plot_pruning_progress,
)

# -----------------------------------------------------------------------------
# Config and globals
# -----------------------------------------------------------------------------
CONFIG_PATH = os.path.join(SCRIPT_DIR, "bayesian_config.json")
MAX_STEPS = 10
MIN_STEPS = 5
N_KL = 500
N_DO = 300


def _encode_alarm_data_to_int(train_df, evaluate_df):
    """Encode object columns to integer codes (same order for train and eval)."""
    train_df = train_df.copy()
    evaluate_df = evaluate_df.copy()
    for col in train_df.columns:
        if train_df[col].dtype == object or train_df[col].dtype.name == "category":
            categories = sorted(train_df[col].dropna().unique())
            cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
            train_df[col] = train_df[col].astype(cat_type).cat.codes
            evaluate_df[col] = evaluate_df[col].astype(cat_type).cat.codes
    return train_df, evaluate_df


def run_alarm():
    """Load ALARM model, generate data, run all pruning methods, return models and histories."""
    print("\n" + "=" * 60)
    print("ALARM MODEL")
    print("=" * 60)
    true_model = get_example_model("alarm")
    n_data, n_train, n_eval = 4200, 900, 900
    data = true_model.simulate(n_samples=n_data, show_progress=False)
    train_data = true_model.simulate(n_samples=n_train, show_progress=False)
    evaluate_data = true_model.simulate(n_samples=n_eval, show_progress=False)
    train_data, evaluate_data = _encode_alarm_data_to_int(train_data, evaluate_data)

    alarm_model = _refit_model(
        list(true_model.edges()),
        list(true_model.nodes()),
        data,
    )
    ev.align_state_names_from_true(true_model, alarm_model)

    max_steps = MAX_STEPS
    ll_fn = ev.evaluate_log_likelihood
    kl_fn = ev.evaluate_kl_divergence
    struct_fn = ev.evaluate_structural_error

    print("\n--- Wavelet pruning ---")
    wavelet_model, wavelet_history = pruning_l2_wavelet(
        true_model, alarm_model,
        data=train_data,
        evaluate_data=evaluate_data,
        max_steps=max_steps,
        evaluate_log_likelihood=ll_fn,
        evaluate_kl_divergence=kl_fn,
        evaluate_structural_error=struct_fn,
    )
    print("\n--- BIC pruning ---")
    bic_model, bic_history = score_pruning(
        true_model, alarm_model,
        data=train_data,
        evaluate_data=evaluate_data,
        score_fn="bic-d",
        max_steps=max_steps,
        min_steps=MIN_STEPS,
        evaluate_log_likelihood=ll_fn,
        evaluate_kl_divergence=kl_fn,
        evaluate_structural_error=struct_fn,
    )
    print("\n--- AIC pruning ---")
    aic_model, aic_history = score_pruning(
        true_model, alarm_model,
        data=train_data,
        evaluate_data=evaluate_data,
        score_fn="aic-d",
        max_steps=max_steps,
        min_steps=MIN_STEPS,
        evaluate_log_likelihood=ll_fn,
        evaluate_kl_divergence=kl_fn,
        evaluate_structural_error=struct_fn,
    )
    print("\n--- BDs pruning ---")
    bds_model, bds_history = score_pruning(
        true_model, alarm_model,
        data=train_data,
        evaluate_data=evaluate_data,
        score_fn="bds",
        max_steps=max_steps,
        min_steps=MIN_STEPS,
        evaluate_log_likelihood=ll_fn,
        evaluate_kl_divergence=kl_fn,
        evaluate_structural_error=struct_fn,
    )
    print("\n--- CSI / structural-error pruning ---")
    csi_model, csi_history = structural_error_pruning(
        true_model, alarm_model,
        data=train_data,
        evaluate_data=evaluate_data,
        max_steps=max_steps,
        evaluate_log_likelihood=ll_fn,
        evaluate_kl_divergence=kl_fn,
        evaluate_structural_error=struct_fn,
    )

    return {
        "true_model": true_model,
        "models": {
            "Wavelet": wavelet_model,
            "BIC": bic_model,
            "AIC": aic_model,
            "BDs": bds_model,
            "CSI": csi_model,
        },
        "histories": {
            "Wavelet": wavelet_history,
            "BIC": bic_history,
            "AIC": aic_history,
            "BDs": bds_history,
            "CSI": csi_history,
        },
        "train_data": train_data,
        "evaluate_data": evaluate_data,
    }


def load_synthetic_model_and_data(config_path):
    """Load synthetic BN from config, sample data, train/eval split."""
    with open(config_path, "r") as f:
        config = json.load(f)
    edges = [tuple(e) for e in config["edges"]]
    model = DiscreteBayesianNetwork(edges)
    cpds = []
    for var, cpd_info in config["cpds"].items():
        cpd = TabularCPD(
            variable=var,
            variable_card=config["variable_card"][var],
            values=cpd_info["values"],
            evidence=cpd_info.get("evidence", []),
            evidence_card=cpd_info.get("evidence_card", []),
        )
        cpds.append(cpd)
    model.add_cpds(*cpds)
    assert model.check_model(), "Synthetic model is invalid"
    sampler = BayesianModelSampling(model)
    data = sampler.forward_sample(size=4000)
    train_data, evaluate_data = train_test_split(data, train_size=0.7, random_state=42)
    return model, train_data, evaluate_data


def run_synthetic():
    """Load synthetic model from config, run all pruning methods, return models and histories."""
    print("\n" + "=" * 60)
    print("SYNTHETIC MODEL")
    print("=" * 60)
    true_model, train_data, evaluate_data = load_synthetic_model_and_data(CONFIG_PATH)
    synthetic_model = _refit_model(
        list(true_model.edges()),
        list(true_model.nodes()),
        train_data,
    )
    ev.align_state_names_from_true(true_model, synthetic_model)

    max_steps = MAX_STEPS
    ll_fn = ev.evaluate_log_likelihood
    kl_fn = ev.evaluate_kl_divergence
    struct_fn = ev.evaluate_structural_error

    print("\n--- Wavelet pruning ---")
    wavelet_model, wavelet_history = pruning_l2_wavelet(
        true_model, synthetic_model,
        data=train_data,
        evaluate_data=evaluate_data,
        max_steps=max_steps,
        evaluate_log_likelihood=ll_fn,
        evaluate_kl_divergence=kl_fn,
        evaluate_structural_error=struct_fn,
    )
    print("\n--- BIC pruning ---")
    bic_model, bic_history = score_pruning(
        true_model, synthetic_model,
        data=train_data,
        evaluate_data=evaluate_data,
        score_fn="bic-d",
        max_steps=max_steps,
        min_steps=MIN_STEPS,
        evaluate_log_likelihood=ll_fn,
        evaluate_kl_divergence=kl_fn,
        evaluate_structural_error=struct_fn,
    )
    print("\n--- AIC pruning ---")
    aic_model, aic_history = score_pruning(
        true_model, synthetic_model,
        data=train_data,
        evaluate_data=evaluate_data,
        score_fn="aic-d",
        max_steps=max_steps,
        min_steps=MIN_STEPS,
        evaluate_log_likelihood=ll_fn,
        evaluate_kl_divergence=kl_fn,
        evaluate_structural_error=struct_fn,
    )
    print("\n--- BDs pruning ---")
    bds_model, bds_history = score_pruning(
        true_model, synthetic_model,
        data=train_data,
        evaluate_data=evaluate_data,
        score_fn="bds",
        max_steps=max_steps,
        min_steps=MIN_STEPS,
        evaluate_log_likelihood=ll_fn,
        evaluate_kl_divergence=kl_fn,
        evaluate_structural_error=struct_fn,
    )
    print("\n--- CSI / structural-error pruning ---")
    csi_model, csi_history = structural_error_pruning(
        true_model, synthetic_model,
        data=train_data,
        evaluate_data=evaluate_data,
        max_steps=max_steps,
        evaluate_log_likelihood=ll_fn,
        evaluate_kl_divergence=kl_fn,
        evaluate_structural_error=struct_fn,
    )

    return {
        "true_model": true_model,
        "models": {
            "Wavelet": wavelet_model,
            "BIC": bic_model,
            "AIC": aic_model,
            "BDs": bds_model,
            "CSI": csi_model,
        },
        "histories": {
            "Wavelet": wavelet_history,
            "BIC": bic_history,
            "AIC": aic_history,
            "BDs": bds_history,
            "CSI": csi_history,
        },
        "train_data": train_data,
        "evaluate_data": evaluate_data,
    }


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="pgmpy")
    print("Current Trends in Industrial Math — Pruning comparison (ALARM + Synthetic)")
    print("Working directory:", os.getcwd())
    print("Project root:", SCRIPT_DIR)

    # 1) ALARM
    alarm_results = run_alarm()
    print("\n" + "=" * 60)
    print("ALARM — Comparison table")
    print("=" * 60)
    run_and_print_comparison(
        true_model=alarm_results["true_model"],
        wavelet_model=alarm_results["models"]["Wavelet"],
        BIC_model=alarm_results["models"]["BIC"],
        AIC_model=alarm_results["models"]["AIC"],
        BDs_model=alarm_results["models"]["BDs"],
        csi_model=alarm_results["models"]["CSI"],
        data=alarm_results["train_data"],
        evaluate_data=alarm_results["evaluate_data"],
        n_kl=N_KL,
        n_do=N_DO,
        evaluation_module=ev,
    )
    fig_alarm, _ = plot_pruning_progress(alarm_results["histories"])
    if fig_alarm is not None:
        out_alarm = os.path.join(SCRIPT_DIR, "alarm_progress.png")
        fig_alarm.savefig(out_alarm, dpi=150, bbox_inches="tight")
        print("\nSaved:", out_alarm)

    # 2) Synthetic (optional if config missing)
    if not os.path.isfile(CONFIG_PATH):
        print("\nSkipping synthetic experiment (bayesian_config.json not found).")
        print("Done. Figure: alarm_progress.png")
        return alarm_results, None

    synthetic_results = run_synthetic()
    print("\n" + "=" * 60)
    print("SYNTHETIC — Comparison table")
    print("=" * 60)
    run_and_print_comparison(
        true_model=synthetic_results["true_model"],
        wavelet_model=synthetic_results["models"]["Wavelet"],
        BIC_model=synthetic_results["models"]["BIC"],
        AIC_model=synthetic_results["models"]["AIC"],
        BDs_model=synthetic_results["models"]["BDs"],
        csi_model=synthetic_results["models"]["CSI"],
        data=synthetic_results["train_data"],
        evaluate_data=synthetic_results["evaluate_data"],
        n_kl=N_KL,
        n_do=N_DO,
        evaluation_module=ev,
    )
    fig_synth, _ = plot_pruning_progress(synthetic_results["histories"])
    if fig_synth is not None:
        out_synth = os.path.join(SCRIPT_DIR, "synthetic_progress.png")
        fig_synth.savefig(out_synth, dpi=150, bbox_inches="tight")
        print("\nSaved:", out_synth)
    print("\nDone. Figures: alarm_progress.png, synthetic_progress.png")
    return alarm_results, synthetic_results


if __name__ == "__main__":
    main()
