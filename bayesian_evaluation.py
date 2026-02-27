"""
Evaluation helpers for Bayesian network pruning: log-likelihood, KL divergence,
structural error, predictive quality, and causal quality (interventions, ACE, colliders).
"""

import copy
import random
import itertools
import numpy as np
import pandas as pd
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


def align_state_names_from_true(
    true_model: DiscreteBayesianNetwork,
    learned_model: DiscreteBayesianNetwork,
) -> None:
    """
    Reorder learned model CPDs so each variable's state index order matches the true model.
    In-place on learned_model. Required for fair KL: after MLE refit, pgmpy may assign
    state order from data (e.g. alphabetical), so the same state string can map to
    different indices in true vs learned, causing huge KL if not aligned.
    """
    for node in list(learned_model.nodes()):
        if node not in true_model.nodes():
            continue
        true_cpd = true_model.get_cpds(node)
        learned_cpd = learned_model.get_cpds(node)
        if true_cpd is None or learned_cpd is None:
            continue
        true_names = getattr(true_cpd, "state_names", None)
        learned_names = getattr(learned_cpd, "state_names", None)
        if not true_names or not learned_names:
            continue
        variables = list(learned_cpd.variables)
        if variables != list(true_cpd.variables):
            continue
        # Build permutation per axis: new_values[t0,t1,...] = learned_values[l0,l1,...]
        # where learned_names[v][li] == true_names[v][ti] => li = learned_names[v].index(true_names[v][ti])
        indices_per_axis = []
        for v in variables:
            if v not in true_names or v not in learned_names:
                break
            true_order = list(true_names[v])
            learned_order = list(learned_names[v])
            if set(true_order) != set(learned_order) or len(true_order) != len(learned_order):
                break
            try:
                indices_per_axis.append([learned_order.index(s) for s in true_order])
            except ValueError:
                break
        if len(indices_per_axis) != len(variables):
            continue
        vals = np.asarray(learned_cpd.values, dtype=float).copy()
        for axis, inds in enumerate(indices_per_axis):
            vals = np.take(vals, inds, axis=axis)
        # Replace CPD with one using true state names and reordered values
        evidence = variables[1:] if len(variables) > 1 else None
        state_names_cpd = {v: list(true_names[v]) for v in variables if v in true_names}
        var_card = len(true_names[variables[0]])
        ev_cards = [len(true_names[v]) for v in variables[1:]] if evidence else None
        new_cpd = TabularCPD(
            node,
            var_card,
            vals,
            evidence=evidence,
            evidence_card=ev_cards,
            state_names=state_names_cpd,
        )
        learned_model.add_cpds(new_cpd)


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
    if data is None or len(data) == 0:
        return 0.0
    total_ll = log_likelihood_score(learned_model, data)
    return -float(total_ll) / len(data)


def evaluate_kl_divergence(
    true_model: DiscreteBayesianNetwork,
    learned_model: DiscreteBayesianNetwork,
    n_samples: int = 1000,
    eps: float = 1e-10,
    verbose: bool = True,
) -> float:
    """
    Monte-Carlo estimate of KL(true || learned) = E_true[ log P_true(X) - log P_learned(X) ].
    Samples are drawn from the true model.
    """
    sampler = BayesianModelSampling(true_model)
    samples = sampler.forward_sample(size=n_samples, show_progress=False)

    log_diffs = []
    skipped = 0
    skip_reasons = {}

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
            log_diffs.append(logp_true - logp_learned)
        except Exception as e:
            skipped += 1
            msg = str(e)
            skip_reasons[msg] = skip_reasons.get(msg, 0) + 1
            if verbose and skipped <= 5:
                print(f"Skipped sample: {e}")

    used = len(log_diffs)
    if verbose:
        print(f"KL used {used}/{n_samples} samples; skipped {skipped}")
    if used < n_samples and skipped / n_samples > 0.05 and verbose:
        print("WARNING: >5% of KL samples skipped (model/state mismatch or missing CPDs?).")
        for reason, cnt in sorted(skip_reasons.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  ({cnt}x) {reason}")

    return float(np.mean(log_diffs)) if used > 0 else 0.0


def generate_ci_tests(nodes, max_cond_set=2, max_tests=500, random_seed=42):
    """
    Generate (X, Y, Z) conditional independence tests.
    Z is the conditioning set with size <= max_cond_set.
    """
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
    """
    Fraction of CI tests where learned and true structure disagree (d-separation).
    """
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


# ---------------------------------------------------------------------------
# Predictive quality: generalization, likelihood, target prediction accuracy
# ---------------------------------------------------------------------------

def evaluate_target_prediction_accuracy(
    model: DiscreteBayesianNetwork,
    data: pd.DataFrame,
    target_var: str,
    evidence_vars: list = None,
    eps: float = 1e-10,
) -> float:
    """
    Predictive quality: fraction of rows where the model's mode prediction for
    target_var (given evidence_vars) equals the actual value.
    """
    if target_var not in model.nodes() or data is None or len(data) == 0:
        return 0.0
    if evidence_vars is None:
        evidence_vars = [c for c in data.columns if c != target_var and c in model.nodes()]
    try:
        infer = VariableElimination(model)
    except Exception:
        return 0.0

    correct = 0
    for _, row in data.iterrows():
        try:
            evidence = {v: row[v] for v in evidence_vars if v in row and v in model.nodes()}
            if not evidence:
                continue
            query = infer.query(variables=[target_var], evidence=evidence)
            probs = np.asarray(query.values).flatten()
            pred_idx = int(np.argmax(probs))
            cpd = model.get_cpds(target_var)
            if hasattr(cpd, "state_names") and target_var in getattr(cpd, "state_names", {}):
                pred_val = list(cpd.state_names[target_var])[pred_idx]
            else:
                pred_val = pred_idx
            actual = row[target_var]
            if actual == pred_val or (isinstance(actual, (int, np.integer)) and actual == pred_idx):
                correct += 1
        except Exception:
            continue
    n = len(data)
    return correct / n if n else 0.0


# ---------------------------------------------------------------------------
# Causal quality: interventional distributions, ACE, colliders
# ---------------------------------------------------------------------------

def interventional_model(model: DiscreteBayesianNetwork, intervention: dict) -> DiscreteBayesianNetwork:
    """
    Build a copy of the model with do(intervention): remove incoming edges to
    intervened variables and set their CPDs to point masses at the given values.
    """
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
    """
    Causal quality: stability of interventional distribution.
    KL( P_true(·|do) || P_learned(·|do) ) by sampling from true interventional. Lower is better.
    """
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


def causal_effect_ace(model: DiscreteBayesianNetwork, X: str, Y: str, x_low, x_high) -> float:
    """
    Average causal effect: E[Y|do(X=x_high)] - E[Y|do(X=x_low)] (Y as state index).
    """
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
        e_high = np.sum(np.arange(n) * p_high)
        e_low = np.sum(np.arange(n) * p_low)
        return float(e_high - e_low)
    except Exception:
        return float("nan")


def list_colliders(model: DiscreteBayesianNetwork) -> list:
    """Nodes with more than one parent (collider structure)."""
    return [n for n in model.nodes() if len(list(model.get_parents(n))) >= 2]


def evaluate_collider_preservation(
    true_model: DiscreteBayesianNetwork,
    learned_model: DiscreteBayesianNetwork,
) -> dict:
    """
    Causal quality: collider preservation. Returns precision, recall, f1.
    """
    true_c = set(list_colliders(true_model))
    learned_c = set(list_colliders(learned_model))
    n_true, n_learned = len(true_c), len(learned_c)
    tp = len(true_c & learned_c)
    precision = tp / n_learned if n_learned else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"n_true_colliders": n_true, "n_learned_colliders": n_learned, "precision": precision, "recall": recall, "f1": f1}
