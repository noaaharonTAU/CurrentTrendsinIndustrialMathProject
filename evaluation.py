"""
Evaluation helpers for Bayesian network pruning: log-likelihood, KL divergence,
structural error, predictive quality, and causal quality (interventions, ACE, colliders).
"""

import random
import itertools
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.metrics import log_likelihood_score
from pgmpy.inference import VariableElimination


def _state_index(cpd, var, value):
    """Get integer index for var's value from CPD (handles string state names or int)."""
    if hasattr(cpd, "state_names") and var in getattr(cpd, "state_names", {}):
        names = cpd.state_names[var]
        if value in names:
            return list(names).index(value)
        try:
            idx = int(value)
            if 0 <= idx < len(names):
                return idx
        except (TypeError, ValueError):
            pass
        return 0
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return 0
    card = cpd.variable_card
    if hasattr(cpd, "cardinality") and cpd.variables and var in cpd.variables:
        if isinstance(cpd.cardinality, (list, tuple)) and len(cpd.cardinality) > cpd.variables.index(var):
            card = cpd.cardinality[cpd.variables.index(var)]
        elif isinstance(cpd.cardinality, dict):
            card = cpd.cardinality.get(var, cpd.variable_card)
    return max(0, min(idx, card - 1))


def _logp_single_var(cpd, var, evidence, eps=1e-10):
    """Log P(var = evidence[var] | parents) from CPD given full evidence dict."""
    parents = list(cpd.variables[1:])
    state_idx = _state_index(cpd, var, evidence[var])
    if len(parents) == 0:
        vals = np.asarray(cpd.values).flatten()
        return np.log(float(vals[state_idx]) + eps)
    parent_evidence = [(p, evidence[p]) for p in parents]
    reduced = cpd.reduce(parent_evidence, inplace=False)
    ridx = _state_index(reduced, var, evidence[var])
    vals = np.asarray(reduced.values).flatten()
    return np.log(float(vals[ridx]) + eps)


def evaluate_log_likelihood(learned_model: DiscreteBayesianNetwork, data: pd.DataFrame) -> float:
    """
    Average negative log-likelihood per row.
    Higher is worse (more negative log-probability).
    """
    n = len(data)
    if n < 500:
        print(f"[Note] Log-likelihood calculated with low n ({n}). Results may have high variance.")
    if data is None or len(data) == 0:
        return 0.0
    total_ll = log_likelihood_score(learned_model, data)
    return -float(total_ll) / n


def evaluate_kl_divergence(
    true_model: DiscreteBayesianNetwork,
    learned_model: DiscreteBayesianNetwork,
    n_samples: int = 1000,
    eps: float = 1e-10,
    verbose: bool = True,
) -> float:
    """
    Monte-Carlo estimate of KL(true || learned) = E_true[ log P_true(X) - log P_learned(X) ].
    Returns NaN if node sets differ or learned is missing CPDs.
    """
    if set(true_model.nodes()) != set(learned_model.nodes()):
        missing = sorted(set(true_model.nodes()) - set(learned_model.nodes()))
        extra = sorted(set(learned_model.nodes()) - set(true_model.nodes()))
        print(f"[BAD KL] Node sets differ. Missing={len(missing)}, Extra={len(extra)}")
        return float("nan")

    missing_cpds = [v for v in true_model.nodes() if learned_model.get_cpds(v) is None]
    if missing_cpds:
        print(f"[BAD KL] learned_model missing CPDs for {len(missing_cpds)} nodes (first 10): {missing_cpds[:10]}")
        return float("nan")

    sampler = BayesianModelSampling(true_model)
    samples = sampler.forward_sample(size=n_samples, show_progress=False)

    log_diffs = []
    skipped = 0

    for _, row in samples.iterrows():
        evidence = row.to_dict()
        try:
            logp_true = sum(
                _logp_single_var(true_model.get_cpds(var), var, evidence, eps)
                for var in true_model.nodes()
            )
            logp_learned = sum(
                _logp_single_var(learned_model.get_cpds(var), var, evidence, eps)
                for var in learned_model.nodes()
            )
            k = len(list(learned_model.nodes()))
            floor = k * np.log(eps)
            if logp_learned <= 0.9 * floor:
                print(f"[BAD KL] learned logP near epsilon floor. logp={logp_learned:.2f}, floor≈{floor:.2f}, k={k}")
                return float("nan")
            log_diffs.append(logp_true - logp_learned)
        except Exception as e:
            skipped += 1
            if verbose and skipped <= 3:
                print(f"[KL] Skipped sample: {type(e).__name__}: {e}")
            continue

    return float(np.mean(log_diffs)) if log_diffs else float("nan")


def align_state_names_from_true(true_model, learned_model):
    """
    Force learned_model CPDs to use the exact same state_names ordering as true_model.
    Returns learned_model (in-place modified) for chaining.
    """
    ref = {}
    for v in true_model.nodes():
        cpd_t = true_model.get_cpds(v)
        if cpd_t is None:
            continue
        if hasattr(cpd_t, "state_names") and cpd_t.state_names:
            if v in cpd_t.state_names:
                ref[v] = list(cpd_t.state_names[v])
            for u, states in cpd_t.state_names.items():
                if u not in ref:
                    ref[u] = list(states)

    for cpd_l in learned_model.get_cpds():
        if cpd_l is None:
            continue
        if not hasattr(cpd_l, "state_names") or cpd_l.state_names is None:
            cpd_l.state_names = {}
        for var in cpd_l.variables:
            if var in ref:
                cpd_l.state_names[var] = list(ref[var])

    return learned_model


def generate_ci_tests(nodes, max_cond_set=2, max_tests=500, random_seed=42):
    """Generate (X, Y, Z) conditional independence tests. Z is the conditioning set with size <= max_cond_set."""
    random.seed(random_seed)
    nodes = list(nodes)
    if len(nodes) < 2:
        return []
    tests = []
    for X, Y in itertools.combinations(nodes, 2):
        others = [n for n in nodes if n not in (X, Y)]
        for k in range(min(max_cond_set, len(others)) + 1):
            for Z in itertools.combinations(others, k):
                tests.append((X, Y, set(Z)))
    random.shuffle(tests)
    return tests[:max_tests]


def evaluate_structural_error(
    true_model: DiscreteBayesianNetwork,
    learned_model: DiscreteBayesianNetwork,
    max_cond_set: int = 2,
    max_tests: int = 500,
    random_seed: int = 42,
) -> float:
    """Fraction of CI tests where learned and true structure disagree (d-separation)."""
    if set(true_model.nodes()) != set(learned_model.nodes()):
        missing = sorted(set(true_model.nodes()) - set(learned_model.nodes()))
        extra = sorted(set(learned_model.nodes()) - set(true_model.nodes()))
        print(f"[BAD STRUCT] Node sets differ for CI tests. Missing={len(missing)}, Extra={len(extra)}")
        return float("nan")

    ci_tests = generate_ci_tests(
        learned_model.nodes(),
        max_cond_set=max_cond_set,
        max_tests=max_tests,
        random_seed=random_seed,
    )
    if not ci_tests:
        return 0.0

    errors = 0
    for X, Y, Z in ci_tests:
        true_sep = not true_model.is_dconnected(X, Y, Z)
        learned_sep = not learned_model.is_dconnected(X, Y, Z)
        if true_sep != learned_sep:
            errors += 1
    return errors / len(ci_tests)


def evaluate_target_prediction_accuracy(
    model: DiscreteBayesianNetwork,
    data: pd.DataFrame,
    target_var: str,
    evidence_vars: list = None,
    eps: float = 1e-10,
) -> float:
    """
    Predictive quality: fraction of usable rows where the model's mode prediction for
    target_var (given evidence_vars) equals the actual value.
    """
    if model is None or data is None or len(data) == 0:
        return float("nan")
    if target_var not in model.nodes():
        return float("nan")
    if evidence_vars is None:
        evidence_vars = model.get_parents(target_var)

    try:
        infer = VariableElimination(model)
    except Exception:
        print("[BAD ACC] VariableElimination failed; cannot compute target accuracy.")
        return float("nan")

    correct = 0
    used = 0

    for _, row in data.iterrows():
        try:
            evidence = {v: row[v] for v in evidence_vars if v in data.columns and v in model.nodes()}
            if not evidence:
                continue
            query = infer.query(variables=[target_var], evidence=evidence)
            probs = np.asarray(query.values).flatten()
            pred_idx = int(np.argmax(probs))
            cpd = model.get_cpds(target_var)
            if cpd is None:
                print(f"[BAD ACC] Missing CPD for target {target_var}.")
                return float("nan")
            if hasattr(cpd, "state_names") and target_var in getattr(cpd, "state_names", {}):
                states = list(cpd.state_names[target_var])
                if pred_idx >= len(states):
                    continue
                pred_val = states[pred_idx]
            else:
                pred_val = pred_idx
            actual = row[target_var]
            used += 1
            if actual == pred_val or (isinstance(actual, (int, np.integer)) and actual == pred_idx):
                correct += 1
        except Exception:
            continue

    if used == 0:
        print("[BAD ACC] No usable rows (empty evidence or inference failures).")
        return float("nan")
    return correct / used


def interventional_model(model: DiscreteBayesianNetwork, intervention: dict) -> DiscreteBayesianNetwork:
    """Build a copy of the model with do(intervention): remove incoming edges to intervened variables."""
    edges = list(model.edges())
    nodes = list(model.nodes())
    intervened_set = set(intervention.keys())
    new_edges = [e for e in edges if e[1] not in intervened_set]
    m = DiscreteBayesianNetwork(new_edges)
    m.add_nodes_from(nodes)

    for node in nodes:
        cpd_old = model.get_cpds(node)
        if cpd_old is None:
            continue
        if node in intervened_set:
            val = intervention[node]
            idx = _state_index(cpd_old, node, val)
            card = cpd_old.variable_card
            vals = np.zeros((card, 1))
            vals[idx, 0] = 1.0
            state_names = getattr(cpd_old, "state_names", None)
            if state_names and node in state_names:
                cpd_new = TabularCPD(node, card, vals, state_names={node: list(state_names[node])})
            else:
                cpd_new = TabularCPD(node, card, vals)
            m.add_cpds(cpd_new)
        else:
            m.add_cpds(cpd_old.copy())
    return m


def evaluate_interventional_kl(
    true_model: DiscreteBayesianNetwork,
    learned_model: DiscreteBayesianNetwork,
    intervention: dict,
    n_samples: int = 500,
    eps: float = 1e-10,
    verbose: bool = False,
) -> float:
    """Causal quality: KL( P_true(·|do) || P_learned(·|do) ) by sampling from true interventional. Lower is better."""
    try:
        true_do = interventional_model(true_model, intervention)
        learned_do = interventional_model(learned_model, intervention)
    except Exception as e:
        if verbose:
            print(f"Interventional model failed: {e}")
        return float("nan")
    sampler = BayesianModelSampling(true_do)
    samples = sampler.forward_sample(size=n_samples, show_progress=False)
    log_diffs = []
    for _, row in samples.iterrows():
        ev = row.to_dict()
        try:
            logp_true = sum(_logp_single_var(true_do.get_cpds(var), var, ev, eps) for var in true_do.nodes())
            logp_learned = sum(_logp_single_var(learned_do.get_cpds(var), var, ev, eps) for var in learned_do.nodes())
            log_diffs.append(logp_true - logp_learned)
        except Exception:
            continue
    return float(np.mean(log_diffs)) if log_diffs else float("nan")


def _states_to_numeric(states):
    """Map state list to numeric indices 0, 1, ... for ACE expectation."""
    return np.arange(len(states))


def causal_effect_ace(model: DiscreteBayesianNetwork, X: str, Y: str, x_low, x_high) -> float:
    """Average causal effect: E[Y|do(X=x_high)] - E[Y|do(X=x_low)] (Y as state index)."""
    if X not in model.nodes() or Y not in model.nodes():
        return float("nan")
    try:
        infer_high = VariableElimination(interventional_model(model, {X: x_high}))
        infer_low = VariableElimination(interventional_model(model, {X: x_low}))
        marg_high = infer_high.query(variables=[Y])
        marg_low = infer_low.query(variables=[Y])
        p_high = np.asarray(marg_high.values).flatten()
        p_low = np.asarray(marg_low.values).flatten()
        n = max(len(p_high), len(p_low))
        p_high = np.resize(p_high, n)
        p_low = np.resize(p_low, n)
        cpd_y = model.get_cpds(Y)
        if hasattr(cpd_y, "state_names") and Y in cpd_y.state_names:
            numeric_vals = _states_to_numeric(cpd_y.state_names[Y])
        else:
            numeric_vals = np.arange(n)
        e_high = np.sum(numeric_vals * p_high)
        e_low = np.sum(numeric_vals * p_low)
        return float(e_high - e_low)
    except Exception:
        return float("nan")


def evaluate_global_ace_difference(
    true_model: DiscreteBayesianNetwork,
    learned_model: DiscreteBayesianNetwork,
) -> float:
    """Mean absolute difference in ACE over all (X,Y) pairs (X ancestor of Y)."""
    learned_model = align_state_names_from_true(true_model, learned_model)
    ace_diffs = []
    nodes = list(true_model.nodes())
    for Y in nodes:
        ancestors = nx.ancestors(true_model, Y)
        for X in ancestors:
            if X not in learned_model.nodes() or Y not in learned_model.nodes():
                continue
            cpd_x = true_model.get_cpds(X)
            if cpd_x is None:
                continue
            x_card = cpd_x.variable_card
            x_states = list(range(x_card))
            for x_low, x_high in itertools.combinations(x_states, 2):
                try:
                    ace_true = causal_effect_ace(true_model, X, Y, x_low, x_high)
                    ace_learned = causal_effect_ace(learned_model, X, Y, x_low, x_high)
                    if not np.isnan(ace_true) and not np.isnan(ace_learned):
                        ace_diffs.append(abs(ace_true - ace_learned))
                except Exception:
                    continue
    if not ace_diffs:
        return np.nan
    return float(np.mean(ace_diffs))


def list_colliders(model: DiscreteBayesianNetwork) -> list:
    """Nodes with more than one parent (collider structure)."""
    return [n for n in model.nodes() if len(list(model.get_parents(n))) >= 2]


def evaluate_collider_preservation(
    true_model: DiscreteBayesianNetwork,
    learned_model: DiscreteBayesianNetwork,
) -> dict:
    """Causal quality: collider preservation. Returns n_true_colliders, n_learned_colliders, recall."""
    true_c = set(list_colliders(true_model))
    learned_c = set(list_colliders(learned_model))
    n_true, n_learned = len(true_c), len(learned_c)
    tp = len(true_c & learned_c)
    recall = tp / n_true if n_true else 0.0
    return {"n_true_colliders": n_true, "n_learned_colliders": n_learned, "recall": recall}


def build_pruning_row_extra(
    true_m,
    learned_m,
    ll_fn,
    kl_fn,
    struct_fn,
    evaluate_data,
    step_edges=None,
    target_var=None,
    pred_fn=None,
    collider_fn=None,
    interventions=None,
    do_kl_fn=None,
    ace_fn=None,
):
    """Build the extra metrics dict for one pruning step. On failure, metrics are set to np.nan."""
    def _safe(f, default=np.nan):
        try:
            return f()
        except Exception:
            return default

    row = {
        "ll_score": _safe(lambda: ll_fn(learned_m, evaluate_data), default=0.0),
        "kl_score": 0.0 if (step_edges is not None and step_edges == 0) else _safe(lambda: kl_fn(true_m, learned_m)),
        "structure_score": _safe(lambda: struct_fn(true_m, learned_m)),
    }
    if target_var and pred_fn and target_var in learned_m.nodes():
        row["pred_accuracy"] = _safe(lambda: pred_fn(learned_m, evaluate_data, target_var))
    else:
        row["pred_accuracy"] = None
    if collider_fn:
        try:
            coll = collider_fn(true_m, learned_m)
            row["collider_recall"] = coll["recall"]
        except Exception:
            row["collider_recall"] = np.nan
    else:
        row["collider_recall"] = None
    if interventions and do_kl_fn:
        kls = []
        for do_dict in interventions:
            try:
                kl = do_kl_fn(true_m, learned_m, do_dict, verbose=False)
                if kl is not None and not (isinstance(kl, float) and np.isnan(kl)):
                    kls.append(kl)
            except Exception as e:
                warnings.warn("Interventional KL failed for {}: {}.".format(do_dict, e), UserWarning, stacklevel=2)
        row["interventional_kl_mean"] = float(np.mean(kls)) if kls else np.nan
    else:
        row["interventional_kl_mean"] = None
    if ace_fn:
        row["global_ace_diff"] = _safe(lambda: ace_fn(true_m, learned_m))
    else:
        row["global_ace_diff"] = None
    return row


def make_row_extra(g, evaluate_data, target_var=None, interventions=None):
    """
    Return a function (true_m, learned_m, step_edges=None) -> metrics dict.
    Reads evaluation functions from the caller's globals `g`.
    """
    ll_fn = g.get("evaluate_log_likelihood")
    kl_fn = g.get("evaluate_kl_divergence")
    struct_fn = g.get("evaluate_structural_error")
    pred_fn = g.get("evaluate_target_prediction_accuracy")
    collider_fn = g.get("evaluate_collider_preservation")
    do_kl_fn = g.get("evaluate_interventional_kl")
    ace_fn = g.get("evaluate_global_ace_difference")
    if target_var is None:
        target_var = g.get("target_var")
    if interventions is None:
        interventions = g.get("interventions")
    interventions = interventions if isinstance(interventions, list) else []
    if ll_fn is None or kl_fn is None or struct_fn is None:
        raise ValueError(
            "evaluate_log_likelihood, evaluate_kl_divergence, evaluate_structural_error "
            "must be in scope or passed via globals"
        )

    def row_extra(true_m, learned_m, step_edges=None):
        return build_pruning_row_extra(
            true_m, learned_m, ll_fn, kl_fn, struct_fn, evaluate_data,
            step_edges=step_edges, target_var=target_var, pred_fn=pred_fn,
            collider_fn=collider_fn, interventions=interventions, do_kl_fn=do_kl_fn, ace_fn=ace_fn,
        )
    return row_extra
