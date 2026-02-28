"""
Load synthetic Bayesian network and data from config JSON (train / eval) for pruning experiments.
Optionally builds a learned-from-noisy-data model for pruning (HillClimbSearch).
"""

import json
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import HillClimbSearch
from sklearn.model_selection import train_test_split

from . import config as config_module
from .helpers import add_gaussian_index_noise, _refit_model


def load_synthetic_from_config(config_path: str = None):
    """
    Load synthetic model from JSON config, generate train/eval data.
    Returns:
        true_synthetic_model, synthetic_model (same as true if no structure learning),
        train_synthetic_data, evaluate_synthetic_data, target_var, interventions
    """
    config_path = config_path or getattr(config_module, "CONFIG_PATH", "bayesian_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    edges = [tuple(e) for e in config["edges"]]
    synthetic_model = DiscreteBayesianNetwork(edges)
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
    synthetic_model.add_cpds(*cpds)
    true_synthetic_model = synthetic_model

    size = getattr(config_module, "SYNTHETIC_SAMPLE_SIZE", 4000)
    ratio = getattr(config_module, "SYNTHETIC_TRAIN_RATIO", 0.7)
    seed = getattr(config_module, "RANDOM_STATE", 42)

    sampler = BayesianModelSampling(true_synthetic_model)
    synthetic_data = sampler.forward_sample(size=size)
    train_synthetic_data, evaluate_synthetic_data = train_test_split(
        synthetic_data, test_size=1 - ratio, random_state=seed
    )

    assert true_synthetic_model.check_model(), "Model is invalid"

    # Optionally build a learned model from noisy data (for pruning experiments)
    n_small = getattr(config_module, "SYNTHETIC_STRUCTURE_SAMPLES", 10)
    eps = getattr(config_module, "SYNTHETIC_NOISE_EPS", 0.2)
    if n_small > 0:
        small_data = sampler.forward_sample(size=n_small)
        noisy_data = add_gaussian_index_noise(small_data, eps=eps, seed=seed)
        noisy_data = noisy_data.round().astype(int)
        for col in noisy_data.columns:
            states = sorted(noisy_data[col].unique())
            mapping = {s: i for i, s in enumerate(states)}
            noisy_data[col] = noisy_data[col].map(mapping)
        for col in noisy_data.columns:
            k = noisy_data[col].nunique()
            if k < 2:
                print(f"[BAD DATA] {col} has only {k} unique value(s) in structure-learning data (n={len(noisy_data)}).")
        if len(noisy_data) < 200:
            print(f"[NOTE] Structure learning with n={len(noisy_data)} is very high-variance; learned DAG may be unreliable.")
        hc = HillClimbSearch(noisy_data)
        best_dag = hc.estimate(scoring_method="bdeu", max_indegree=10)
        synthetic_model = _refit_model(best_dag.edges, best_dag.nodes, noisy_data)
    else:
        synthetic_model = _refit_model(
            list(true_synthetic_model.edges()),
            list(true_synthetic_model.nodes()),
            train_synthetic_data,
        )

    target_var = getattr(config_module, "SYNTHETIC_TARGET_VAR", None)
    interventions = getattr(config_module, "SYNTHETIC_INTERVENTIONS", [])

    return true_synthetic_model, synthetic_model, train_synthetic_data, evaluate_synthetic_data, target_var, interventions
