"""
CSI (Context-Specific Independence) / structural-error based pruning for Bayesian networks.
Iteratively prunes the edge with lowest average KL (most CSI-irrelevant).
"""

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from . import config as config_module
from .helpers import _refit_model, warn_if_bad_cpds
from .evaluation import make_row_extra


def _cpd_shape(cpd: TabularCPD):
    """Return shape of CPD values in variable order (child, parent1, parent2, ...)."""
    nvars = len(cpd.variables)
    if hasattr(cpd, "cardinality") and cpd.cardinality is not None:
        if isinstance(cpd.cardinality, (list, tuple)):
            if len(cpd.cardinality) >= nvars:
                return tuple(cpd.cardinality[:nvars])
            return tuple(cpd.cardinality) + (cpd.variable_card,) * (nvars - len(cpd.cardinality))
        if isinstance(cpd.cardinality, dict):
            return tuple(cpd.cardinality.get(v, cpd.variable_card) for v in cpd.variables)
    ev_card = getattr(cpd, "evidence_card", None)
    if isinstance(ev_card, (list, tuple)):
        return (cpd.variable_card,) + tuple(ev_card)
    return (cpd.variable_card,) + (cpd.variable_card,) * max(0, nvars - 1)


def kl_divergence_csi(p, q, epsilon=1e-10):
    """KL divergence between two probability distributions p and q. Epsilon prevents log(0) errors."""
    p = np.asarray(p, dtype=float).flatten() + epsilon
    q = np.asarray(q, dtype=float).flatten() + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def is_parent_csi_irrelevant_approx(cpd: TabularCPD, parent: str, epsilon: float = 0.03):
    """
    Check whether a parent is CSI-irrelevant for the child in some contexts.
    Returns (is_irrelevant_in_some_context, list_of_irrelevant_contexts).
    """
    parents = list(cpd.variables[1:])
    if parent not in parents:
        return True, []

    parent_idx = parents.index(parent)
    shape = _cpd_shape(cpd)
    parent_card = shape[parent_idx + 1]
    values = np.asarray(cpd.values, dtype=float).reshape(shape)

    other_parents = [p for p in parents if p != parent]
    other_axes = [i + 1 for i, p in enumerate(parents) if p != parent]
    other_cards = [shape[i + 1] for i, p in enumerate(parents) if p != parent]
    irrelevant_contexts = []

    if not other_axes:
        dists = []
        for y in range(parent_card):
            slices = [slice(None)] * values.ndim
            slices[parent_idx + 1] = y
            dists.append(values[tuple(slices)].flatten())
        context_irrelevant = True
        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                if kl_divergence_csi(dists[i], dists[j]) > epsilon:
                    context_irrelevant = False
                    break
            if not context_irrelevant:
                break
        if context_irrelevant:
            irrelevant_contexts.append({})
        return len(irrelevant_contexts) > 0, irrelevant_contexts

    for context in np.ndindex(*other_cards):
        slices = [slice(None)] * values.ndim
        for ax, val in zip(other_axes, context):
            slices[ax] = val
        dists = []
        for y in range(parent_card):
            slices[parent_idx + 1] = y
            dists.append(values[tuple(slices)].flatten())
        context_irrelevant = True
        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                if kl_divergence_csi(dists[i], dists[j]) > epsilon:
                    context_irrelevant = False
                    break
            if not context_irrelevant:
                break
        if context_irrelevant:
            irrelevant_contexts.append(dict(zip(other_parents, context)))

    return len(irrelevant_contexts) > 0, irrelevant_contexts


def compute_avg_kl(cpd: TabularCPD, parent: str) -> float:
    """
    Average KL divergence across all parent-value pairs, over all contexts.
    Lower = parent is more irrelevant (good candidate for removal).
    """
    parents = list(cpd.variables[1:])
    if parent not in parents:
        return 0.0

    parent_idx = parents.index(parent)
    values = np.asarray(cpd.values, dtype=float)
    shape = values.shape
    parent_card = shape[parent_idx + 1]
    other_axes = [i + 1 for i, p in enumerate(parents) if p != parent]
    other_cards = [shape[i + 1] for i, p in enumerate(parents) if p != parent]

    total_kl = 0.0
    num_pairs = 0
    contexts = list(np.ndindex(*other_cards)) if other_axes else [()]

    for context in contexts:
        slices = [slice(None)] * values.ndim
        for ax, val in zip(other_axes, context):
            slices[ax] = val
        dists = []
        for y in range(parent_card):
            s = slices.copy()
            s[parent_idx + 1] = y
            dists.append(values[tuple(s)].flatten())
        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                total_kl += kl_divergence_csi(dists[i], dists[j])
                num_pairs += 1

    return total_kl / num_pairs if num_pairs > 0 else 0.0


def structural_error_pruning(
    true_model,
    pruned_model,
    data,
    evaluate_data,
    evaluate_log_likelihood=None,
    evaluate_kl_divergence=None,
    evaluate_structural_error=None,
    target_var=None,
    interventions=None,
    evaluate_target_prediction_accuracy=None,
    evaluate_collider_preservation=None,
    evaluate_interventional_kl=None,
    evaluate_global_ace_difference=None,
):
    """
    Iteratively prune the edge with lowest average KL (most CSI-irrelevant).
    Uses _refit_model (fit + clamp) from helpers.
    """
    from . import evaluation as ev
    g = {k: getattr(ev, k) for k in dir(ev) if not k.startswith("_")}
    row_extra = make_row_extra(g, evaluate_data, target_var=target_var, interventions=interventions)

    max_steps = getattr(config_module, "max_steps", 10)

    pruned_model = _refit_model(
        list(pruned_model.edges()),
        list(pruned_model.nodes()),
        data,
    )
    warn_if_bad_cpds(pruned_model)

    baseline_extra = row_extra(true_model, pruned_model, step_edges=0)
    history = [{
        "step": 0,
        "removed_edges": [],
        "num_edges": len(pruned_model.edges()),
        "score_name": "Structural Error",
        "score": 0.0,
        **baseline_extra,
    }]

    all_pruned_edges = []

    for step in range(1, max_steps + 1):
        edges_info = []
        for child in pruned_model.nodes():
            cpd = pruned_model.get_cpds(child)
            if cpd is None:
                continue
            for parent in list(pruned_model.get_parents(child)):
                try:
                    avg_kl = compute_avg_kl(cpd, parent)
                    edges_info.append(((parent, child), avg_kl))
                except Exception as e:
                    print(f"[BAD] compute_avg_kl failed for edge {parent}->{child}: {e}")
                    continue

        bad_cpds = [n for n in pruned_model.nodes() if pruned_model.get_cpds(n) is None]
        if bad_cpds:
            print(f"[BAD] Missing CPDs for {len(bad_cpds)} nodes (first 10): {bad_cpds[:10]}")
        if not edges_info:
            print("No edges left to prune; stopping.")
            break

        edges_info.sort(key=lambda x: x[1])
        best_edge, best_kl = edges_info[0]

        print(f"\nSTEP {step}: removing edge {best_edge} with avg KL={best_kl:.5f}")

        new_edges = [e for e in pruned_model.edges() if e != best_edge]
        nodes = list(pruned_model.nodes())
        all_pruned_edges.append(best_edge)

        pruned_model = _refit_model(new_edges, nodes, data)

        step_extra = row_extra(true_model, pruned_model)
        history.append({
            "step": step,
            "edge": best_edge,
            "num_edges": len(pruned_model.edges()),
            "score_name": "Structural Error",
            "score": best_kl,
            **step_extra,
        })

    print("\nFinal remaining edges:")
    print(list(pruned_model.edges()))
    print("\nAll pruned edges:")
    print(all_pruned_edges)
    return pruned_model, history
