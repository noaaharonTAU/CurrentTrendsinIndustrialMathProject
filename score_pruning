def evaluate_single_edge_deletions(current_model: DiscreteBayesianNetwork, data: pd.DataFrame, score_fn="bic-d", verbose=True):
    """Evaluates removing each edge based on given score function."""
    nodes = list(current_model.nodes())
    base_edges = list(current_model.edges())
    candidates = []

    for edge in base_edges:
        try:
            new_edges = [e for e in base_edges if e != edge]
            pruned = fit_model_with_smoothing(new_edges, nodes, data)
            score = structure_score(pruned, data, scoring_method=score_fn)

            candidates.append({
                "op": "remove",
                "edge": edge,
                "score": score,
                "model": pruned
            })
        except Exception as e:
            if verbose:
                print(f"Skipping edge {edge}: {e}")

    return candidates

def score_pruning(true_model: DiscreteBayesianNetwork, model: DiscreteBayesianNetwork, data: pd.DataFrame, evaluate_data:pd.DataFrame, score_fn="bic-d"):
    """
    Greedy edge pruning based on a given score function.
    Stops when score no longer improves, but runs at least `min_steps`.
    """
    score_name = score_fn

    # -------------------------
    # Fit baseline
    # -------------------------
    current_score = structure_score(model, data, scoring_method=f"{score_fn}")
    pruned_model = model
    print(f"Baseline train {score_name}: {current_score:.3f}")

    history = []
    # Step 0: baseline row
    baseline_row = {
        "step": 0,
        "removed_edges": [],
        "num_edges": len(model.edges()),
        "score_name": score_name,
        "score": current_score,
        "ll_score": evaluate_log_likelihood(pruned_model, evaluate_data),
        "kl_score": 0,
        "structure_score": evaluate_structural_error(true_model, pruned_model),
    }

    # Initialize history with baseline
    history = [baseline_row]

    print("\nStarting iterative pruning...")
    for step in range(1, max_steps + 1):
        # -------------------------
        # Evaluate single-edge deletions
        # -------------------------
        candidates = evaluate_single_edge_deletions(pruned_model, data, score_fn, verbose=False)
        if not candidates:
            print("No valid candidates left; stopping.")
            break

        # -------------------------
        # Select best candidate
        # -------------------------
        best = max(candidates, key=lambda x: x["score"])

        # -------------------------
        # Apply pruning first
        # -------------------------
        pruned_model = best["model"]
        previous_score = current_score
        current_score = best["score"]

        print(f"\nSTEP {step}: removed edge {best['edge']}")
        print(f"  train {score_name} = {current_score:.6f} | delta = {current_score - previous_score:+.6f}")

        # -------------------------
        # Log history
        # -------------------------
        history.append({
            "step": step,
            "edge": best["edge"],
            "num_edges": len(pruned_model.edges()),
            "score_name": score_name,
            "score": current_score,
            "ll_score": evaluate_log_likelihood(pruned_model, evaluate_data),
            "kl_score": evaluate_kl_divergence(true_model, pruned_model),
            "structure_score": evaluate_structural_error(true_model, pruned_model),

        })

        # -------------------------
        # Stop if no improvement and past minimum steps
        # -------------------------
        if step >= min_steps and (current_score - previous_score) <= 0:
            print(f"No improvement after step {step}; stopping.")
            break

    # -------------------------
    # Final results
    # -------------------------
    print("\nPruning history:")
    display(pd.DataFrame(history))

    print("\nFinal model edges:")
    print(list(pruned_model.edges()))

    return pruned_model, history
