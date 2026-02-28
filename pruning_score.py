"""
Score-based pruning for Bayesian network structure learning.
Iteratively removes the edge whose removal gives the best structure score (e.g. BIC, AIC, BDs).
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.metrics import structure_score

from . import config as config_module
from .helpers import _clamp_cpds, _refit_model, warn_if_bad_cpds
from .evaluation import make_row_extra


def evaluate_single_edge_deletions(
    current_model: DiscreteBayesianNetwork,
    data,
    score_fn: str = "bic-d",
    verbose: bool = False,
):
    """Evaluate removing each edge using the given structure score."""
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
                print(f"[CANDIDATE FAIL] edge {edge}: {type(e).__name__}: {e}")
            continue

    return candidates


def score_pruning(
    true_model,
    pruned_model,
    data,
    evaluate_data,
    score_fn: str = "bic-d",
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
    Iteratively remove the edge whose removal gives the best structure score (e.g. BIC).
    Baseline is fitted and clamped once; each candidate is fitted and clamped.
    """
    from . import evaluation as ev
    g = {k: getattr(ev, k) for k in dir(ev) if not k.startswith("_")}
    row_extra = make_row_extra(g, evaluate_data, target_var=target_var, interventions=interventions)

    min_steps = getattr(config_module, "min_steps", 10)
    max_steps = getattr(config_module, "max_steps", 10)

    pruned_model = _refit_model(
        list(pruned_model.edges()),
        list(pruned_model.nodes()),
        data,
    )
    warn_if_bad_cpds(pruned_model)

    current_score = structure_score(pruned_model, data, scoring_method=score_fn)
    score_name = score_fn
    print(f"Baseline train {score_name}: {current_score:.3f}")

    baseline_extra = row_extra(true_model, pruned_model, step_edges=0)
    history = [{
        "step": 0,
        "removed_edges": [],
        "num_edges": len(pruned_model.edges()),
        "score_name": score_fn,
        "score": current_score,
        **baseline_extra,
    }]

    print("\nStarting iterative pruning...")
    for step in range(1, max_steps + 1):
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

        step_extra = row_extra(true_model, pruned_model)
        history.append({
            "step": step,
            "removed_edges": best["edge"],
            "num_edges": len(pruned_model.edges()),
            "score_name": score_fn,
            "score": current_score,
            **step_extra,
        })

        if step >= min_steps and (current_score - previous_score) <= 0:
            print(f"No improvement after step {step}; stopping.")
            break

    print("\nPruning history:")
    import pandas as pd
    print(pd.DataFrame(history).to_string())
    print("\nFinal model edges:")
    print(list(pruned_model.edges()))
    return pruned_model, history
