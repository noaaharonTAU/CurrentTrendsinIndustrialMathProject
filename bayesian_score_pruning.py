"""
Score-based pruning for Bayesian network structure learning.
Uses _clamp_cpds and _refit_model from bayesian_wavelet_pruning_fixed.
"""

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.metrics import structure_score

from bayesian_wavelet_pruning_fixed import _clamp_cpds, _refit_model

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)


def evaluate_single_edge_deletions(
    current_model: DiscreteBayesianNetwork,
    data: pd.DataFrame,
    score_fn: str = "bic-d",
    verbose: bool = True,
):
    """Evaluate removing each edge using the given structure score. Fitted models are clamped to avoid log(0)."""
    base_edges = list(current_model.edges())
    nodes = list(current_model.nodes())
    candidates = []

    for edge in base_edges:
        try:
            new_edges = [e for e in base_edges if e != edge]
            pruned = DiscreteBayesianNetwork(new_edges)
            pruned.add_nodes_from(nodes)
            pruned.fit(data, estimator=MaximumLikelihoodEstimator)
            _clamp_cpds(pruned, min_prob=1e-10)
            score = structure_score(pruned, data, scoring_method=score_fn)
            candidates.append({
                "op": "remove",
                "edge": edge,
                "score": score,
                "model": pruned,
            })
        except Exception as e:
            if verbose:
                print(f"Skipping edge {edge}: {e}")

    return candidates


def score_pruning(
    true_model,
    model,
    data,
    evaluate_data,
    score_fn: str = "bic-d",
    max_steps=None,
    min_steps=None,
    evaluate_log_likelihood=None,
    evaluate_kl_divergence=None,
    evaluate_structural_error=None,
):
    """
    Iteratively remove the edge whose removal gives the best structure score (e.g. BIC).
    Baseline is fitted and clamped once; each candidate is fitted and clamped.
    Optional args can be omitted and read from global scope in a notebook.
    """
    g = globals()
    steps_max = max_steps if max_steps is not None else g.get("max_steps", 15)
    steps_min = min_steps if min_steps is not None else g.get("min_steps", 5)
    ll_fn = evaluate_log_likelihood or g.get("evaluate_log_likelihood")
    kl_fn = evaluate_kl_divergence or g.get("evaluate_kl_divergence")
    struct_fn = evaluate_structural_error or g.get("evaluate_structural_error")
    if ll_fn is None or kl_fn is None or struct_fn is None:
        raise ValueError(
            "evaluate_log_likelihood, evaluate_kl_divergence, evaluate_structural_error "
            "must be in scope or passed"
        )

    # Fit baseline once (same pattern as wavelet: refit + clamp)
    pruned_model = _refit_model(
        list(model.edges()),
        list(model.nodes()),
        data,
    )
    current_score = structure_score(pruned_model, data, scoring_method=score_fn)
    score_name = score_fn
    print(f"Baseline train {score_name}: {current_score:.3f}")

    history = [{
        "step": 0,
        "removed_edges": [],
        "num_edges": len(pruned_model.edges()),
        "score_name": score_name,
        "score": current_score,
        "ll_score": ll_fn(pruned_model, evaluate_data),
        "kl_score": 0.0,
        "structure_score": struct_fn(true_model, pruned_model),
    }]

    print("\nStarting iterative pruning...")
    for step in range(1, steps_max + 1):
        candidates = evaluate_single_edge_deletions(
            pruned_model, data, score_fn=score_fn, verbose=False
        )
        if not candidates:
            print("No valid candidates left; stopping.")
            break

        best = max(candidates, key=lambda x: x["score"])
        previous_score = current_score
        current_score = best["score"]
        pruned_model = best["model"]

        print(f"\nSTEP {step}: removed edge {best['edge']}")
        print(f"  train {score_name} = {current_score:.6f} | delta = {current_score - previous_score:+.6f}")

        history.append({
            "step": step,
            "edge": best["edge"],
            "num_edges": len(pruned_model.edges()),
            "score_name": score_name,
            "score": current_score,
            "ll_score": ll_fn(pruned_model, evaluate_data),
            "kl_score": kl_fn(true_model, pruned_model),
            "structure_score": struct_fn(true_model, pruned_model),
        })

        if step >= steps_min and (current_score - previous_score) <= 0:
            print(f"No improvement after step {step}; stopping.")
            break

    print("\nPruning history:")
    display(pd.DataFrame(history))

    print("\nFinal model edges:")
    print(list(pruned_model.edges()))
    return pruned_model, history
