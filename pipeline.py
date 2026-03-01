"""
Single-run pipeline and worker entry points for parallel execution.
Workers must live in a non-__main__ module so ProcessPoolExecutor does not re-run main.py.
"""

import io
import logging
import contextlib

# Suppress pgmpy in worker processes too
logging.getLogger("pgmpy").setLevel(logging.ERROR)

from . import config as config_module
from .data_alarm import load_alarm_data
from .data_synthetic import load_synthetic_from_config
from .pruning_wavelet import pruning_l2_wavelet
from .pruning_score import score_pruning
from .pruning_structural import structural_error_pruning
from .comparison import build_comparison_from_last_step


def run_alarm_once(seed=None):
    """Run one full ALARM pipeline; returns (histories_dict, comparison_df). Suppresses stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        true_model, pruned_model, train_data, eval_data, target_var, interventions = load_alarm_data(seed=seed)
        _, wavelet_history = pruning_l2_wavelet(
            true_model=true_model, pruned_model=pruned_model, data=train_data, evaluate_data=eval_data,
            target_var=target_var, interventions=interventions,
        )
        _, BIC_history = score_pruning(
            true_model=true_model, pruned_model=pruned_model, data=train_data, evaluate_data=eval_data,
            score_fn="bic-d", target_var=target_var, interventions=interventions,
        )
        _, AIC_history = score_pruning(
            true_model=true_model, pruned_model=pruned_model, data=train_data, evaluate_data=eval_data,
            score_fn="aic-d", target_var=target_var, interventions=interventions,
        )
        _, BDs_history = score_pruning(
            true_model=true_model, pruned_model=pruned_model, data=train_data, evaluate_data=eval_data,
            score_fn="bds", target_var=target_var, interventions=interventions,
        )
        _, csi_history = structural_error_pruning(
            true_model=true_model, pruned_model=pruned_model, data=train_data, evaluate_data=eval_data,
            target_var=target_var, interventions=interventions,
        )
        histories = {
            "Wavelet": wavelet_history,
            "BIC": BIC_history,
            "AIC": AIC_history,
            "BDs": BDs_history,
            "CSI": csi_history,
        }
        df = build_comparison_from_last_step(histories)
    return histories, df


def run_synthetic_once(config_path=None, seed=None):
    """Run one full synthetic pipeline; returns (histories_dict, comparison_df). Suppresses stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        true_model, pruned_model, train_data, eval_data, target_var, interventions = load_synthetic_from_config(
            config_path=config_path, seed=seed
        )
        _, wavelet_history = pruning_l2_wavelet(
            true_model=true_model, pruned_model=pruned_model, data=train_data, evaluate_data=eval_data,
            target_var=target_var, interventions=interventions,
        )
        _, BIC_history = score_pruning(
            true_model=true_model, pruned_model=pruned_model, data=train_data, evaluate_data=eval_data,
            score_fn="bic-d", target_var=target_var, interventions=interventions,
        )
        _, AIC_history = score_pruning(
            true_model=true_model, pruned_model=pruned_model, data=train_data, evaluate_data=eval_data,
            score_fn="aic-d", target_var=target_var, interventions=interventions,
        )
        _, BDs_history = score_pruning(
            true_model=true_model, pruned_model=pruned_model, data=train_data, evaluate_data=eval_data,
            score_fn="bds", target_var=target_var, interventions=interventions,
        )
        _, csi_history = structural_error_pruning(
            true_model=true_model, pruned_model=pruned_model, data=train_data, evaluate_data=eval_data,
            target_var=target_var, interventions=interventions,
        )
        histories = {
            "Wavelet": wavelet_history,
            "BIC": BIC_history,
            "AIC": AIC_history,
            "BDs": BDs_history,
            "CSI": csi_history,
        }
        df = build_comparison_from_last_step(histories)
    return histories, df


def run_alarm_worker(seed):
    """Entry point for ProcessPoolExecutor: run one ALARM run and return (histories, df)."""
    return run_alarm_once(seed=seed)


def run_synthetic_worker(config_path_seed):
    """Entry point for ProcessPoolExecutor: (config_path, seed) -> (histories, df)."""
    config_path, seed = config_path_seed
    return run_synthetic_once(config_path=config_path, seed=seed)
