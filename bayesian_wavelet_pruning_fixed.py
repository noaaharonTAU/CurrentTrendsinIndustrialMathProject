"""
Fixed Bayesian wavelet pruning helpers for BN structure learning.
Paste this into a single notebook code cell (after imports).
Uses model.fit(data, estimator=MaximumLikelihoodEstimator) for refitting.
"""

import itertools
import copy
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator

try:
    from bayesian_evaluation import align_state_names_from_true
except ImportError:
    def align_state_names_from_true(true_model, learned_model):
        pass


def p_i_given_pi(cpd: TabularCPD, x_i: int, pi_i: dict):
    """Return P(X_i = x_i | Pi_i = pi_i) as a number from the CPD."""
    slicer = [x_i]  # child index
    for p in cpd.variables[1:]:  # parents in CPD order
        val = int(pi_i.get(p, 0))
        card = cpd.cardinality[cpd.variables.index(p)]
        if val >= card:
            raise ValueError(f"Parent {p} state {val} exceeds cardinality {card}")
        slicer.append(val)
    return float(cpd.values[tuple(slicer)])


def p_i_minus_z_fast(cpd, x_i, pi_i, parent_to_remove, parent_z_counts, context_counts):
    """Coarse probability P^{(-Z)}(X_i=x_i | pi_{-Z}) by marginalizing Z. Keys are integer tuples."""
    eps = 1e-6
    parents = cpd.variables[1:]
    remaining_parents = [p for p in parents if p != parent_to_remove]

    key_minus_z = tuple(pi_i[p] for p in remaining_parents)

    N_context = context_counts.get(key_minus_z, 0)
    if N_context == 0:
        return eps

    Z_card = cpd.cardinality[cpd.variables.index(parent_to_remove)]
    p_minus_z = 0.0

    for z_val in range(Z_card):
        key_full = key_minus_z + (z_val,)
        count_z = parent_z_counts.get(key_full, 0)
        P_z_given_context = count_z / N_context

        pi_i_copy = {**pi_i}
        pi_i_copy[parent_to_remove] = z_val
        p_full = p_i_given_pi(cpd, x_i, pi_i_copy)
        p_minus_z += P_z_given_context * p_full

    return p_minus_z if p_minus_z > 0 else eps


def _to_tuple_keys(d: dict) -> dict:
    """Ensure all dict keys are tuples (fixes pandas scalar-key groupby issue)."""
    return {(k,) if not isinstance(k, tuple) else k: v for k, v in d.items()}


def _data_to_index_keys(data: pd.DataFrame, parents: list, cpd: TabularCPD) -> pd.DataFrame:
    """Map parent columns in data to integer indices 0..card-1 so groupby keys match pi_i."""
    out = data[list(parents)].copy()
    for p in parents:
        col = out[p]
        card = cpd.cardinality[cpd.variables.index(p)]
        if hasattr(cpd, "state_names") and p in getattr(cpd, "state_names", {}):
            name_to_idx = {name: i for i, name in enumerate(cpd.state_names[p])}
            out[p] = col.map(lambda x: name_to_idx.get(x, x) if not isinstance(x, (int, np.integer)) else (x if 0 <= x < card else 0))
        else:
            uniq = np.unique(col.dropna())
            out[p] = col.map(dict(zip(uniq, range(len(uniq)))))
    return out


def compute_detail(cpd, parent_to_remove: str, data: pd.DataFrame, debug=False):
    """L2 wavelet detail norm for one parent Z of a child. Uses integer-index keys for robust lookup."""
    eps = 1e-12
    child_card = cpd.variable_card
    parents = cpd.variables[1:]
    parent_cards = [cpd.cardinality[cpd.variables.index(p)] for p in parents]
    remaining_parents = [p for p in parents if p != parent_to_remove]

    # Build count dicts on integer-indexed data so keys match pi_i (integer tuples)
    data_idx = _data_to_index_keys(data, parents, cpd)
    if remaining_parents:
        context_counts = _to_tuple_keys(
            data_idx.groupby(remaining_parents, observed=True).size().to_dict()
        )
        parent_z_counts = _to_tuple_keys(
            data_idx.groupby(remaining_parents, observed=True)[parent_to_remove]
            .value_counts()
            .to_dict()
        )
    else:
        context_counts = {(): len(data)}
        parent_z_counts = _to_tuple_keys(
            data_idx[parent_to_remove].value_counts().to_dict()
        )

    total_N = len(data)
    psi_squared_weighted = 0.0
    total_weight = 0.0

    for pi_vals in itertools.product(*[range(card) for card in parent_cards]):
        pi_i = dict(zip(parents, pi_vals))
        key_minus_z = tuple(pi_i[p] for p in remaining_parents)

        N_context = context_counts.get(key_minus_z, 0)
        if N_context == 0:
            continue

        weight_context = N_context / total_N

        for x_i in range(child_card):
            p_full = p_i_given_pi(cpd, x_i, pi_i)
            p_coarse = p_i_minus_z_fast(
                cpd, x_i, pi_i, parent_to_remove, parent_z_counts, context_counts,
            )
            psi = np.log(p_full + eps) - np.log(p_coarse + eps)
            joint_weight = weight_context * p_full
            psi_squared_weighted += joint_weight * psi**2
            total_weight += joint_weight

    return np.sqrt(psi_squared_weighted / total_weight) if total_weight > 0 else 0.0


def compute_all_wavelet_norms(model: DiscreteBayesianNetwork, data: pd.DataFrame):
    """Iterates over all edges in the network to calculate their L2 wavelet norms."""
    norms = []
    for child in model.nodes():
        cpd = model.get_cpds(child)
        if cpd is None:
            continue
        for parent in model.get_parents(child):
            try:
                l2_norm = compute_detail(cpd, parent, data)
                norms.append(((parent, child), l2_norm))
            except Exception:
                continue
    return norms


def compute_tau(model: DiscreteBayesianNetwork, data: pd.DataFrame, k=1):
    """Threshold τ that would prune exactly k edges from sorted wavelet norms."""
    all_norms = []
    for child in model.nodes():
        cpd = model.get_cpds(child)
        if cpd is None:
            continue
        for parent in model.get_parents(child):
            try:
                l2_norm = compute_detail(cpd, parent, data)
                all_norms.append(l2_norm)
            except Exception:
                continue
    if len(all_norms) <= k:
        return None
    sorted_norms = np.sort(np.array(all_norms))
    return float(sorted_norms[k])


def _clamp_cpds(model, min_prob=1e-10):
    """Clamp CPD values to min_prob and renormalize so log-likelihood never sees log(0)."""
    for cpd in model.get_cpds():
        vals = np.clip(cpd.values, min_prob, None)
        vals = vals / vals.sum(axis=0, keepdims=True)
        cpd.values = vals


def _refit_model(edges, nodes, data):
    """Build and fit a DiscreteBayesianNetwork with MLE, then clamp CPDs to avoid log(0)."""
    m = DiscreteBayesianNetwork(edges)
    m.add_nodes_from(nodes)
    m.fit(data, estimator=MaximumLikelihoodEstimator)
    _clamp_cpds(m, min_prob=1e-10)
    return m


def pruning_l2_wavelet(true_model, pruned_model, evaluate_data, data, max_steps=None,
                       evaluate_log_likelihood=None, evaluate_kl_divergence=None, evaluate_structural_error=None,
                       target_var=None, interventions=None,
                       evaluate_target_prediction_accuracy=None, evaluate_collider_preservation=None, evaluate_interventional_kl=None):
    """
    Iteratively prune edges with smallest L2 wavelet norm.
    Records predictive (ll, kl, pred_accuracy) and causal (structure_score, collider_*, interventional_kl) metrics.
    Optional: target_var + evaluate_target_prediction_accuracy for pred_accuracy;
              interventions + evaluate_interventional_kl for interventional_kl_mean.
    """
    g = globals()
    ll_fn = evaluate_log_likelihood or g.get("evaluate_log_likelihood")
    kl_fn = evaluate_kl_divergence or g.get("evaluate_kl_divergence")
    struct_fn = evaluate_structural_error or g.get("evaluate_structural_error")
    pred_fn = evaluate_target_prediction_accuracy or g.get("evaluate_target_prediction_accuracy")
    collider_fn = evaluate_collider_preservation or g.get("evaluate_collider_preservation")
    do_kl_fn = evaluate_interventional_kl or g.get("evaluate_interventional_kl")
    steps = max_steps if max_steps is not None else g.get("max_steps", 15)
    if ll_fn is None or kl_fn is None or struct_fn is None:
        raise ValueError("evaluate_log_likelihood, evaluate_kl_divergence, evaluate_structural_error must be in scope or passed")

    def _row_extra(true_m, learned_m, step_edges=None):
        row = {
            "ll_score": ll_fn(learned_m, evaluate_data),
            "kl_score": 0.0 if (step_edges is not None and step_edges == 0) else kl_fn(true_m, learned_m),
            "structure_score": struct_fn(true_m, learned_m),
        }
        if target_var and pred_fn and target_var in learned_m.nodes():
            row["pred_accuracy"] = pred_fn(learned_m, evaluate_data, target_var)
        else:
            row["pred_accuracy"] = None
        if collider_fn:
            coll = collider_fn(true_m, learned_m)
            row["collider_recall"] = coll["recall"]
            row["collider_precision"] = coll["precision"]
        else:
            row["collider_recall"] = None
            row["collider_precision"] = None
        if interventions and do_kl_fn:
            kls = []
            for do_dict in interventions:
                kl = do_kl_fn(true_m, learned_m, do_dict, n_samples=300, verbose=False)
                if kl is not None and not (isinstance(kl, float) and np.isnan(kl)):
                    kls.append(kl)
            row["interventional_kl_mean"] = float(np.mean(kls)) if kls else None
        else:
            row["interventional_kl_mean"] = None
        return row

    pruned_model = copy.deepcopy(pruned_model)
    pruned_model = _refit_model(
        list(pruned_model.edges()),
        list(pruned_model.nodes()),
        data,
    )
    align_state_names_from_true(true_model, pruned_model)

    k_per_step = 1
    baseline_extra = _row_extra(true_model, pruned_model, step_edges=0)
    history = [{
        "step": 0,
        "removed_edges": [],
        "num_edges": len(pruned_model.edges()),
        "score_name": "wavelet_l2",
        "score": 0.0,
        **baseline_extra,
    }]

    for step in range(1, steps + 1):
        norms_with_edges = compute_all_wavelet_norms(pruned_model, data)
        if len(norms_with_edges) <= k_per_step:
            print(f"\nSTEP {step}: Not enough edges to prune {k_per_step}; stopping.")
            break

        norms_with_edges.sort(key=lambda x: x[1])
        edges_to_remove = [e for e, _ in norms_with_edges[:k_per_step]]

        for (parent, child), val in norms_with_edges[:k_per_step]:
            print(f"\nSTEP {step}:  Removing edge {parent}->{child} (L2={val:.5f})")

        new_edges = [e for e in pruned_model.edges() if e not in edges_to_remove]
        all_nodes = list(pruned_model.nodes())

        pruned_model = DiscreteBayesianNetwork(new_edges)
        for node in all_nodes:
            if node not in pruned_model.nodes():
                pruned_model.add_node(node)

        pruned_model = _refit_model(
            list(pruned_model.edges()),
            list(pruned_model.nodes()),
            data,
        )
        align_state_names_from_true(true_model, pruned_model)

        remaining_norms = compute_all_wavelet_norms(pruned_model, data)
        if remaining_norms:
            print(sorted(remaining_norms, key=lambda x: x[1])[:10])
        mean_wavelet_l2 = np.mean([v for _, v in remaining_norms]) if remaining_norms else 0.0

        step_extra = _row_extra(true_model, pruned_model)
        history.append({
            "step": step,
            "edge": edges_to_remove,
            "num_edges": len(pruned_model.edges()),
            "score_name": "wavelet_l2",
            "score": mean_wavelet_l2,
            **step_extra,
        })

    print("\nFinal pruned edges:", list(pruned_model.edges()))
    return pruned_model, history


# ---------------------------------------------------------------------------
# NOTEBOOK: Use the same call as before; optional args for full evaluation:
#   from bayesian_evaluation import (
#       evaluate_log_likelihood, evaluate_kl_divergence, evaluate_structural_error,
#       evaluate_target_prediction_accuracy, evaluate_collider_preservation, evaluate_interventional_kl,
#   )
#   wavelet_model, wavelet_history = pruning_l2_wavelet(
#       true_model=..., pruned_model=..., evaluate_data=..., data=...,
#       target_var="MyTarget",  # optional: for pred_accuracy
#       interventions=[{"X": 0}, {"X": 1}],  # optional: for interventional_kl_mean
#   )
# History rows include: ll_score, kl_score, structure_score, pred_accuracy,
#                       collider_recall, collider_precision, interventional_kl_mean.
# ---------------------------------------------------------------------------
