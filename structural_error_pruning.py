def kl_divergence_csi(p, q, epsilon=1e-10):
    """
    KL divergence between two probability distributions p and q.
    Epsilon prevents log(0) errors.
    """
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def is_parent_csi_irrelevant_approx(cpd, parent, epsilon=0.03):
    """
    Checks whether a parent is CSI-irrelevant for a child node.
    Returns (is_irrelevant_in_some_context, list_of_irrelevant_contexts).

    A parent is considered fully removable only if it is irrelevant
    in ALL contexts (checked in pruning_removing_csi_approx).
    """
    parents = list(cpd.variables[1:])

    if parent not in parents:
        return True, []

    parent_idx = parents.index(parent)
    parent_card = cpd.cardinality[parent_idx + 1]

    values = cpd.values.reshape(cpd.cardinality)

    other_parents = [p for p in parents if p != parent]
    other_axes = [i + 1 for i, p in enumerate(parents) if p != parent]
    other_cards = [cpd.cardinality[i + 1] for i, p in enumerate(parents) if p != parent]

    irrelevant_contexts = []

    # If there are no other parents, there is only one context
    if not other_axes:
        dists = []
        for y in range(parent_card):
            slices = [slice(None)] * values.ndim
            slices[parent_idx + 1] = y
            dists.append(values[tuple(slices)])

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

    # Loop over every combination of other parent values (each is one context)
    for context in np.ndindex(*other_cards):
        slices = [slice(None)] * values.ndim
        for ax, val in zip(other_axes, context):
            slices[ax] = val

        dists = []
        for y in range(parent_card):
            slices[parent_idx + 1] = y
            dists.append(values[tuple(slices)])

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

def compute_avg_kl(cpd, parent):
    """
    Computes the average KL divergence across all parent value pairs,
    averaged over all contexts. Lower = more irrelevant.
    """
    parents = list(cpd.variables[1:])
    parent_idx = parents.index(parent)
    parent_card = cpd.cardinality[parent_idx + 1]
    values = cpd.values.reshape(cpd.cardinality)

    other_axes = [i + 1 for i, p in enumerate(parents) if p != parent]
    other_cards = [cpd.cardinality[i + 1] for i, p in enumerate(parents) if p != parent]

    total_kl = 0.0
    num_contexts = 0

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
            dists.append(values[tuple(slices)])

        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                total_kl += kl_divergence_csi(dists[i], dists[j])
                num_contexts += 1

    return total_kl / num_contexts if num_contexts > 0 else 0.0

def structural_error_pruning(true_model: DiscreteBayesianNetwork, model: DiscreteBayesianNetwork, evaluate_data:pd.DataFrame,  data: pd.DataFrame):
    """Iteratively prunes edges from a Bayesian Network using approximate CSI."""

    history = []
    all_pruned_edges = []
    current_model = model

    # Step 0: baseline row
    baseline_row = {
        "step": 0,
        "removed_edges": [],
        "num_edges": len(model.edges()),
        "score_name": "Structural Error",
        "score": 0,
        "ll_score": evaluate_log_likelihood(current_model, evaluate_data),
        "kl_score": 0,
        "structure_score": evaluate_structural_error(true_model, current_model),
    }

    # Initialize history with baseline
    history = [baseline_row]

    for step in range(1, max_steps + 1):
        edges_info = []

        for child in current_model.nodes():
            cpd = current_model.get_cpds(child)
            if cpd is None:
                continue

            for parent in current_model.get_parents(child):
                avg_kl = compute_avg_kl(cpd, parent)
                edges_info.append(((parent, child), avg_kl))

        if not edges_info:
            print("No edges left to prune; stopping.")
            break

        # Sort edges by avg KL divergence (lowest = most irrelevant)
        edges_info.sort(key=lambda x: x[1])
        best_edge, best_kl = edges_info[0]  # pick the least relevant edge

        print(f"\nSTEP {step}: removing edge {best_edge} with avg KL={best_kl:.5f}")
        current_model.remove_edge(*best_edge)
        all_pruned_edges.append(best_edge)

        # Refit after pruning
        current_model = fit_model_with_smoothing(
            list(current_model.edges()),
            list(current_model.nodes()),
            data
        )

        # Log step
        history.append({
            "step": step,
            "edge": [best_edge],
            "num_edges": len(list(current_model.edges())),
            "score_name": "Structural Error",
            "score": best_kl,
            "ll_score": evaluate_log_likelihood(current_model, evaluate_data),
            "kl_score": evaluate_kl_divergence(true_model, current_model),
            "structure_score": evaluate_structural_error(true_model, current_model),
        })

    print("\nFinal remaining edges:")
    print(list(current_model.edges()))
    print("\nAll pruned edges:")
    print(all_pruned_edges)

    return current_model, history
