"""
Helpers: smoothing models, add noise to data.
Used by pruning methods and data loaders.
"""

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator


def add_gaussian_index_noise(df: pd.DataFrame, eps: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """Add index-based noise: with probability eps, shift each value to a neighboring state (mod cardinality)."""
    rng = np.random.default_rng(seed)
    noisy = df.copy()

    for col in noisy.columns:
        states = np.sort(noisy[col].unique())
        k = len(states)
        idx = {s: i for i, s in enumerate(states)}

        mask = rng.random(len(noisy)) < eps
        for i in noisy[mask].index:
            cur = idx[noisy.at[i, col]]
            shift = rng.choice([-1, 1])
            noisy.at[i, col] = states[(cur + shift) % k]

    return noisy


def _clamp_cpds(model: DiscreteBayesianNetwork, min_prob: float = 1e-10) -> None:
    """Clamp CPD values to min_prob and renormalize so log-likelihood never sees log(0)."""
    for cpd in model.get_cpds():
        vals = np.clip(cpd.values, min_prob, None)
        vals = vals / vals.sum(axis=0, keepdims=True)
        cpd.values = vals


def _refit_model(edges, nodes, data: pd.DataFrame) -> DiscreteBayesianNetwork:
    """Build and fit a DiscreteBayesianNetwork with MLE, then clamp CPDs to avoid log(0)."""
    m = DiscreteBayesianNetwork(edges)
    m.add_nodes_from(nodes)
    m.fit(data, estimator=MaximumLikelihoodEstimator)
    _clamp_cpds(m, min_prob=1e-10)
    return m


def warn_if_bad_cpds(model: DiscreteBayesianNetwork, atol: float = 1e-6) -> None:
    """Print warnings if any CPD has NaN/Inf or columns not normalized to 1."""
    for cpd in model.get_cpds():
        vals = np.asarray(cpd.values, dtype=float)

        if not np.isfinite(vals).all():
            print(f"[BAD CPD] {cpd.variable}: has NaN/Inf values")
            return

        vcard = cpd.variable_card
        mat = vals.reshape(vcard, -1)
        col_sums = mat.sum(axis=0)

        if not np.allclose(col_sums, 1.0, atol=atol):
            worst = float(np.max(np.abs(col_sums - 1.0)))
            print(f"[BAD CPD] {cpd.variable}: columns not normalized (max |sum-1| = {worst:.2e})")
            return
