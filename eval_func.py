# Evaluation function: log-likelihood
def evaluate_log_likelihood(learned_model: DiscreteBayesianNetwork, data: pd.DataFrame) -> float:
  """
  Average negative log-likelihood per row.
  """
  total_ll = log_likelihood_score(learned_model, data)
  return -float(total_ll) / len(data)

# Evaluation function: kl_divergence
def evaluate_kl_divergence(true_model: DiscreteBayesianNetwork, learned_model: DiscreteBayesianNetwork):
   """
   Computes KL divergence between true_model and the current (pruned) model.
   """
   n_samples = 1000

   # Sample from the TRUE MODEL (not the pruned one)
   sampler = BayesianModelSampling(true_model)
   samples = sampler.forward_sample(size=n_samples, show_progress=False)


   log_diffs = []
   for _, row in samples.iterrows():
       evidence = row.to_dict()

       try:
           # Compute log probability under TRUE model
           logp_true = 0.0
           for var in true_model.nodes():
               cpd_true = true_model.get_cpds(var)
               parents_true = list(true_model.get_parents(var))


               if len(parents_true) == 0:
                   # P(var)
                   state_idx = list(cpd_true.state_names[var]).index(evidence[var])
                   logp_true += np.log(cpd_true.values[state_idx] + 1e-10)
               else:
                   # P(var | parents)
                   parent_evidence = [(p, evidence[p]) for p in parents_true]
                   reduced = cpd_true.reduce(parent_evidence, inplace=False)
                   state_idx = list(cpd_true.state_names[var]).index(evidence[var])
                   logp_true += np.log(reduced.values.flatten()[state_idx] + 1e-10)


           # Compute log probability under PRUNED model
           logp_pruned = 0.0
           for var in learned_model.nodes():
               cpd_pruned = learned_model.get_cpds(var)
               parents_pruned = list(learned_model.get_parents(var))


               if len(parents_pruned) == 0:
                   state_idx = list(cpd_pruned.state_names[var]).index(evidence[var])
                   logp_pruned += np.log(cpd_pruned.values[state_idx] + 1e-10)
               else:
                   parent_evidence = [(p, evidence[p]) for p in parents_pruned]
                   reduced = cpd_pruned.reduce(parent_evidence, inplace=False)
                   state_idx = list(cpd_pruned.state_names[var]).index(evidence[var])
                   logp_pruned += np.log(reduced.values.flatten()[state_idx] + 1e-10)


           # KL divergence: E[log P - log Q]
           log_diffs.append(logp_true - logp_pruned)


       except Exception as e:
           # Skip samples that cause errors
           continue


   return max(np.mean(log_diffs),0.0) if len(log_diffs) > 0 else 0.0

# for Evaluation function: structural error
def generate_ci_tests(nodes, max_cond_set=2, max_tests=500, random_seed=42):
    """
    Generate (X, Y, Z) conditional independence tests.
    Z is the conditioning set with size <= max_cond_set.
    """
    random.seed(random_seed)
    tests = []
    nodes = list(nodes)

    for X, Y in itertools.combinations(nodes, 2):
        others = [n for n in nodes if n not in (X, Y)]
        for k in range(min(max_cond_set, len(others)) + 1):
            for Z in itertools.combinations(others, k):
                tests.append((X, Y, set(Z)))

    random.shuffle(tests)
    return tests[:max_tests]


# Evaluation function: structural error
def evaluate_structural_error(true_model: DiscreteBayesianNetwork, learned_model: DiscreteBayesianNetwork) -> float:
    """
    Compute the fraction of CI disagreements between learned_model and true_model
    using a precomputed list of CI tests (X, Y, Z).
    """
    errors = 0

    # Generate CI tests internally
    # Adjust max_cond_set and max_tests as needed for performance/accuracy
    ci_tests = generate_ci_tests(learned_model.nodes(), max_cond_set=2, max_tests=500, random_seed=42)

    for X, Y, Z in ci_tests:
        true_sep = not true_model.is_dconnected(X, Y, Z)
        learned_sep = not learned_model.is_dconnected(X, Y, Z)

        if true_sep != learned_sep:
            errors += 1

    return errors / len(ci_tests)
