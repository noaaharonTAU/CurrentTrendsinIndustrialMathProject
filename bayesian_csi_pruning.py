"""
CSI (Context-Specific Independence) / structural-error based pruning for Bayesian networks.
Uses _refit_model from bayesian_wavelet_pruning_fixed for fit+clamp.
"""

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from bayesian_wavelet_pruning_fixed import _refit_model


def _cpd_shape(cpd: TabularCPD, state_names_override: dict = None):
    """
    Return shape of CPD values in variable order (child, parent1, parent2, ...).
    Must satisfy np.prod(shape) == cpd.values.size. Uses evidence_card and variable_card,
    or state_names lengths, and validates against values.size.
    state_names_override: optional dict (node -> list of state names) from the model.
    """
    values = np.asarray(cpd.values)
    total = values.size
    nvars = len(cpd.variables)
    if nvars == 0:
        return tuple(values.shape)

    # 0) Use state_names from model if provided (most reliable after fit)
    if state_names_override and isinstance(state_names_override, dict):
        try:
            shape = tuple(len(state_names_override.get(v, [])) for v in cpd.variables)
            if all(s > 0 for s in shape) and np.prod(shape) == total:
                return shape
        except (KeyError, TypeError):
            pass

    # 1) Prefer (variable_card,) + evidence_card (pgmpy order = variables[0] is child, rest are evidence)
    var_card = getattr(cpd, "variable_card", None)
    ev_card = getattr(cpd, "evidence_card", None)
    if var_card is not None and isinstance(ev_card, (list, tuple)) and len(ev_card) == nvars - 1:
        shape = (var_card,) + tuple(ev_card)
        if np.prod(shape) == total:
            return shape

    # 2) Try cardinality list/dict in variables order
    card = getattr(cpd, "cardinality", None)
    if card is not None:
        if isinstance(card, (list, tuple)) and len(card) >= nvars:
            shape = tuple(card[:nvars])
            if np.prod(shape) == total:
                return shape
        if isinstance(card, dict):
            shape = tuple(card.get(v, getattr(cpd, "variable_card", 2)) for v in cpd.variables)
            if np.prod(shape) == total:
                return shape

    # 3) Use state_names on CPD (order = cpd.variables)
    state_names = getattr(cpd, "state_names", None)
    if state_names and isinstance(state_names, dict):
        try:
            shape = tuple(len(state_names[v]) for v in cpd.variables)
            if np.prod(shape) == total:
                return shape
        except (KeyError, TypeError):
            pass

    # 4) Infer from values.shape: pgmpy stores (variable_card, product(evidence_card))
    if values.ndim == 2 and values.shape[0] * values.shape[1] == total:
        var_card = values.shape[0]
        n_evidence = values.shape[1]
        if nvars == 1:
            return (var_card,)
        if nvars == 2:
            return (var_card, n_evidence)
        # Multiple parents: factor n_evidence into (nvars-1) dimensions; try evidence_card if product matches
        if isinstance(ev_card, (list, tuple)) and len(ev_card) == nvars - 1 and np.prod(ev_card) == n_evidence:
            return (var_card,) + tuple(ev_card)
        # Single possible factorization
        if nvars == 3 and n_evidence > 1:
            for a in range(2, int(n_evidence ** 0.5) + 1):
                if n_evidence % a == 0:
                    b = n_evidence // a
                    return (var_card, a, b)

    # 5) Fallback: 2D only for nvars==2
    if values.ndim >= 1 and total >= 1 and var_card is not None and total % var_card == 0:
        ncol = total // var_card
        if nvars == 1:
            return (var_card,)
        if nvars == 2:
            return (var_card, ncol)

    # Last resort: return actual shape so reshape doesn't change size (may break indexing for nvars>2)
    return tuple(values.shape)


def kl_divergence_csi(p, q, epsilon=1e-10):
    """
    KL divergence between two probability distributions p and q.
    Epsilon prevents log(0) errors.
    """
    p = np.asarray(p, dtype=float).flatten() + epsilon
    q = np.asarray(q, dtype=float).flatten() + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def is_parent_csi_irrelevant_approx(cpd: TabularCPD, parent: str, epsilon: float = 0.03, state_names_override: dict = None):
    """
    Check whether a parent is CSI-irrelevant for the child in some contexts.
    Returns (is_irrelevant_in_some_context, list_of_irrelevant_contexts).
    state_names_override: optional dict from model for correct CPD shape.
    """
    parents = list(cpd.variables[1:])
    if parent not in parents:
        return True, []

    parent_idx = parents.index(parent)
    shape = _cpd_shape(cpd, state_names_override)
    if np.prod(shape) != np.asarray(cpd.values).size:
        return False, []
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


def compute_avg_kl(cpd: TabularCPD, parent: str, state_names_override: dict = None) -> float:
    """
    Average KL divergence across all parent-value pairs, over all contexts.
    Lower = parent is more irrelevant (good candidate for removal).
    state_names_override: optional dict from model for correct CPD shape.
    """
    parents = list(cpd.variables[1:])
    if parent not in parents:
        return 0.0

    parent_idx = parents.index(parent)
    shape = _cpd_shape(cpd, state_names_override)
    if np.prod(shape) != np.asarray(cpd.values).size:
        return float("inf")
    parent_card = shape[parent_idx + 1]
    values = np.asarray(cpd.values, dtype=float).reshape(shape)

    other_axes = [i + 1 for i, p in enumerate(parents) if p != parent]
    other_cards = [shape[i + 1] for i, p in enumerate(parents) if p != parent]

    total_kl = 0.0
    num_pairs = 0

    if not other_axes:
        contexts = [()]
    else:
        contexts = list(np.ndindex(*other_cards))

    for context in contexts:
        slices = [slice(None)] * values.ndim
        for ax, val in zip(other_axes, context):
            slices[ax] = val

        dists = []
        for y in range(parent_card):
            slices[parent_idx + 1] = y
            dists.append(values[tuple(slices)].flatten())

        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                total_kl += kl_divergence_csi(dists[i], dists[j])
                num_pairs += 1

    return total_kl / num_pairs if num_pairs > 0 else 0.0


def structural_error_pruning(
    true_model: DiscreteBayesianNetwork,
    model: DiscreteBayesianNetwork,
    evaluate_data: pd.DataFrame,
    data: pd.DataFrame,
    max_steps=None,
    evaluate_log_likelihood=None,
    evaluate_kl_divergence=None,
    evaluate_structural_error=None,
):
    """
    Iteratively prune the edge with lowest average KL (most CSI-irrelevant).
    Uses _refit_model (fit + clamp) from bayesian_wavelet_pruning_fixed.
    Optional args can be omitted and read from global scope in a notebook.
    """
    g = globals()
    steps_max = max_steps if max_steps is not None else g.get("max_steps", 15)
    ll_fn = evaluate_log_likelihood or g.get("evaluate_log_likelihood")
    kl_fn = evaluate_kl_divergence or g.get("evaluate_kl_divergence")
    struct_fn = evaluate_structural_error or g.get("evaluate_structural_error")
    if ll_fn is None or kl_fn is None or struct_fn is None:
        raise ValueError(
            "evaluate_log_likelihood, evaluate_kl_divergence, evaluate_structural_error "
            "must be in scope or passed"
        )

    # Refit both models once (same pattern as wavelet/score: fit + clamp)
    true_model = _refit_model(
        list(true_model.edges()),
        list(true_model.nodes()),
        data,
    )
    current_model = _refit_model(
        list(model.edges()),
        list(model.nodes()),
        data,
    )

    # State names from model for correct CPD shape (avoids reshape errors after fit)
    state_names = getattr(current_model, "state_names", None)
    if not state_names and hasattr(current_model, "nodes"):
        try:
            state_names = {
                n: list(np.unique(data[n].dropna().astype(str)))
                for n in current_model.nodes()
                if n in data.columns
            }
            if not state_names:
                state_names = None
        except Exception:
            state_names = None

    history = [{
        "step": 0,
        "removed_edges": [],
        "num_edges": len(current_model.edges()),
        "score_name": "Structural Error",
        "score": 0.0,
        "ll_score": ll_fn(current_model, evaluate_data),
        "kl_score": 0.0,
        "structure_score": struct_fn(true_model, current_model),
    }]

    all_pruned_edges = []

    for step in range(1, steps_max + 1):
        edges_info = []
        for child in current_model.nodes():
            cpd = current_model.get_cpds(child)
            if cpd is None:
                continue
            for parent in list(current_model.get_parents(child)):
                try:
                    avg_kl = compute_avg_kl(cpd, parent, state_names_override=state_names)
                    edges_info.append(((parent, child), avg_kl))
                except Exception:
                    continue

        if not edges_info:
            print("No edges left to prune; stopping.")
            break

        edges_info.sort(key=lambda x: x[1])
        best_edge, best_kl = edges_info[0]

        print(f"\nSTEP {step}: removing edge {best_edge} with avg KL={best_kl:.5f}")

        new_edges = [e for e in current_model.edges() if e != best_edge]
        nodes = list(current_model.nodes())
        all_pruned_edges.append(best_edge)

        current_model = _refit_model(new_edges, nodes, data)

        history.append({
            "step": step,
            "edge": [best_edge],
            "num_edges": len(current_model.edges()),
            "score_name": "Structural Error",
            "score": best_kl,
            "ll_score": ll_fn(current_model, evaluate_data),
            "kl_score": kl_fn(true_model, current_model),
            "structure_score": struct_fn(true_model, current_model),
        })

    print("\nFinal remaining edges:")
    print(list(current_model.edges()))
    print("\nAll pruned edges:")
    print(all_pruned_edges)
    return current_model, history
