"""
Load ALARM Bayesian network and data (train / eval) for pruning experiments.
Uses pgmpy's built-in alarm example model.
"""

import pandas as pd
from pgmpy.utils import get_example_model

from . import config as config_module
from .helpers import _refit_model


def load_alarm_data():
    """
    Load true ALARM model, generate train/eval data, and fit initial pruned_model (same structure, MLE on data).
    Returns:
        true_alarm_model, alarm_model, train_alarm_data, evaluate_alarm_data, target_var, interventions
    """
    true_alarm_model = get_example_model("alarm")

    n_data = getattr(config_module, "ALARM_DATA_SAMPLES", 1000)
    n_train = getattr(config_module, "ALARM_TRAIN_SAMPLES", 2000)
    n_eval = getattr(config_module, "ALARM_EVAL_SAMPLES", 2000)
    seed = getattr(config_module, "RANDOM_STATE", 42)

    data = true_alarm_model.simulate(n_samples=n_data, show_progress=False, seed=seed)
    train_alarm_data = true_alarm_model.simulate(n_samples=n_train, show_progress=False, seed=seed + 1)
    evaluate_alarm_data = true_alarm_model.simulate(n_samples=n_eval, show_progress=False, seed=seed + 2)

    alarm_model = _refit_model(
        list(true_alarm_model.edges()),
        list(true_alarm_model.nodes()),
        data,
    )

    # Encode object columns to integer codes for consistency
    for col in train_alarm_data.columns:
        if train_alarm_data[col].dtype == object:
            train_alarm_data[col] = pd.Categorical(train_alarm_data[col]).codes
            evaluate_alarm_data[col] = pd.Categorical(evaluate_alarm_data[col]).codes
    for col in train_alarm_data.columns:
        if train_alarm_data[col].dtype == object:
            categories = sorted(train_alarm_data[col].unique())
            cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
            train_alarm_data[col] = train_alarm_data[col].astype(cat_type).cat.codes
            evaluate_alarm_data[col] = evaluate_alarm_data[col].astype(cat_type).cat.codes

    target_var = getattr(config_module, "ALARM_TARGET_VAR", "CATECHOL")
    interventions = getattr(config_module, "ALARM_INTERVENTIONS", None)
    if interventions is None:
        intervention_nodes = getattr(config_module, "ALARM_INTERVENTION_NODES", [])
        interventions = []
        for node in intervention_nodes:
            if node not in true_alarm_model.nodes():
                continue
            cpd = true_alarm_model.get_cpds(node)
            if cpd is None:
                continue
            for state in range(cpd.variable_card):
                interventions.append({node: state})

    return true_alarm_model, alarm_model, train_alarm_data, evaluate_alarm_data, target_var, interventions
