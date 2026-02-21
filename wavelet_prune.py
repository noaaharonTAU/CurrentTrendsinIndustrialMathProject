def p_i_given_pi(cpd: TabularCPD, x_i: int, pi_i: dict):
    """Return P(X_i = x_i | Pi_i = pi_i) as a number from the CPD."""
    slicer = [x_i]  # child index
    for p in cpd.variables[1:]:  # parents in CPD order
        val = int(pi_i.get(p, 0))
        card = cpd.cardinality[cpd.variables.index(p)]
        if val >= card:
            raise ValueError(f"Parent {p} state {val} exceeds cardinality {card}")
        slicer.append(val)
    return cpd.values[tuple(slicer)]


def p_i_minus_z(cpd: TabularCPD, x_i: int, pi_i: dict, parent_to_remove: str, data: pd.DataFrame):
    """Compute coarse probability P^{(-Z)}(X_i=x_i | pi_{-Z}) by marginalizing Z."""
    parents = cpd.variables[1:]
    remaining_parents = [p for p in parents if p != parent_to_remove]

    # Filter the dataset to the context
    df_context = data.copy()
    for p in remaining_parents:
        df_context = df_context[df_context[p] == pi_i[p]]
    N_context = len(df_context)
    if N_context == 0:
        return 1e-12  # tiny number to avoid log(0)

    Z_card = cpd.cardinality[cpd.variables.index(parent_to_remove)]
    p_minus_z = 0.0

    for z_val in range(Z_card):
        P_z_given_context = np.sum(df_context[parent_to_remove] == z_val) / N_context
        pi_full = pi_i.copy()
        pi_full[parent_to_remove] = z_val
        p_full = p_i_given_pi(cpd, x_i, pi_full)
        p_minus_z += P_z_given_context * p_full

    return p_minus_z


def compute_detail(cpd: TabularCPD, parent_to_remove: str, data: pd.DataFrame):
    """Compute L2 wavelet norm for one parent Z of a child."""
    eps = 1e-12
    child_card = cpd.variable_card
    parents = cpd.variables[1:]

    # Get CPD cardinalities for all parents
    parent_cards = [cpd.cardinality[cpd.variables.index(p)] for p in parents]
    all_parent_assignments = list(itertools.product(*[range(card) for card in parent_cards]))

    psi_squared_weighted = 0.0
    total_weight = 0.0

    for pi_vals in all_parent_assignments:
        pi_i = dict(zip(parents, pi_vals))

        # compute context weight from training data
        df_context = data.copy()
        for p, val in pi_i.items():
            df_context = df_context[df_context[p] == val]
        N_context = len(df_context)
        if N_context == 0:
            continue  # skip impossible contexts
        weight_context = N_context / len(data)

        for x_i in range(child_card):
            p_full = p_i_given_pi(cpd, x_i, pi_i)
            p_coarse = p_i_minus_z(cpd, x_i, pi_i, parent_to_remove, data)
            psi = np.log(p_full + eps) - np.log(p_coarse + eps)
            psi_squared_weighted += weight_context * psi**2
            total_weight += weight_context

    return np.sqrt(psi_squared_weighted / total_weight) if total_weight > 0 else 0.0

def compute_all_wavelet_norms(model: DiscreteBayesianNetwork, data: pd.DataFrame):
    """Iterates over all edges in the network to calculate their L2 wavelet norms."""
    norms = []

    for child in model.nodes():
        cpd = model.get_cpds(child)
        if cpd is None:
            continue

        for parent in model.get_parents(child):
            l2_norm = compute_detail(cpd, parent, data)
            norms.append(((parent, child), l2_norm))

    return norms

def compute_tau(model: DiscreteBayesianNetwork, data: pd.DataFrame, k=1):
    """Determines a threshold τ that would prune exactly k edges based on the sorted wavelet norms."""
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
        return None  # not enough edges to prune k

    all_norms = np.array(all_norms)
    sorted_norms = np.sort(all_norms)

    tau = sorted_norms[k]  # exactly k norms < tau

    return tau


def pruning_l2_wavelet(true_model: DiscreteBayesianNetwork, model: DiscreteBayesianNetwork, evaluate_data:pd.DataFrame, data: pd.DataFrame):
    """Main iterative l2_wavelet pruning routine"""

    pruned_model = fit_model_with_smoothing(
      list(model.edges()),
      list(model.nodes()),
      data)

    k_per_step = 1
    history = []

    # Step 0: baseline row
    baseline_row = {
        "step": 0,
        "removed_edges": [],
        "num_edges": len(model.edges()),
        "score_name": "wavelet_l2",
        "score": min(compute_all_wavelet_norms(pruned_model, evaluate_data)),
        "ll_score": evaluate_log_likelihood(pruned_model, evaluate_data),
        "kl_score": 0,
        "structure_score": evaluate_structural_error(true_model, pruned_model),
    }

    history = [baseline_row]

    for step in range(1, max_steps + 1):
        # ---- Compute all current wavelet norms ----
        norms_with_edges = compute_all_wavelet_norms(pruned_model, data)

        if len(norms_with_edges) <= k_per_step:
            print(f"\nSTEP {step}: Not enough edges to prune {k_per_step}; stopping.")
            break

        # ---- Sort edges by wavelet norm ----
        norms_with_edges.sort(key=lambda x: x[1])

        # ---- Select exactly k weakest edges ----
        edges_to_remove = [edge for edge, _ in norms_with_edges[:k_per_step]]

        tau = norms_with_edges[k_per_step][1]

        for (parent, child), val in norms_with_edges[:k_per_step]:
            print(f"\nSTEP {step}:  Removing edge {parent}->{child} (L2={val:.5f})")

        # ---- Remove edges ----
        for parent, child in edges_to_remove:
            pruned_model.remove_edge(parent, child)

        # ---- Refit model with smoothing ----
        pruned_model = fit_model_with_smoothing(
            list(pruned_model.edges()),
            list(pruned_model.nodes()),
            data
        )

        # ---- Compute mean wavelet norm after pruning ----
        remaining_norms = compute_all_wavelet_norms(pruned_model, data)
        mean_wavelet_l2 = (
            np.mean([v for _, v in remaining_norms])
            if remaining_norms else 0.0
        )

        # ---- Evaluation metrics ----
        history.append({
            "step": step,
            "edge": edges_to_remove,
            "num_edges": len(pruned_model.edges()),
            "score_name": "wavelet_l2",
            "score": mean_wavelet_l2,
            "ll_score": evaluate_log_likelihood(pruned_model, evaluate_data),
            "kl_score": evaluate_kl_divergence(true_model, pruned_model),
            "structure_score": evaluate_structural_error(true_model, pruned_model),
        })

    print("\nFinal pruned edges:", list(pruned_model.edges()))
    return pruned_model, history
