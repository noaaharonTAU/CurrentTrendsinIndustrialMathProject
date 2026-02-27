# Current Trends in Industrial Math Project

Comparison of Bayesian network pruning methods: **Wavelet (L2)**, **BIC**, **AIC**, **BDs**, and **CSI (structural-error)** on the ALARM network and a synthetic model.

## Setup

```bash
git clone https://github.com/noaaharonTAU/CurrentTrendsinIndustrialMathProject.git
cd CurrentTrendsinIndustrialMathProject
pip install -r requirements.txt
```

## Run everything

From the project root:

```bash
python main.py
```

This will:

1. **ALARM model** — Load the ALARM BN (`pgmpy.utils.get_example_model('alarm')`), generate train/eval data, run all five pruning methods, print the comparison table, and save `alarm_progress.png`.
2. **Synthetic model** — If `bayesian_config.json` is present: load the synthetic BN, generate data, run all five methods, print the comparison table, and save `synthetic_progress.png`. If the config file is missing, synthetic is skipped and only ALARM runs.

Runtime: several minutes depending on hardware (ALARM has 37 nodes; synthetic size is defined in the config).

## What you need in the repo

| File | Description |
|------|-------------|
| `main.py` | Entry point (run this). |
| `bayesian_config.json` | **Required for synthetic experiment.** Defines nodes, edges, `variable_card`, and CPDs. Put it in the same directory as `main.py`. If missing, only ALARM runs. |
| `bayesian_evaluation.py` | Evaluation helpers (log-likelihood, KL, structural error, colliders, interventional KL). |
| `bayesian_wavelet_pruning_fixed.py` | Wavelet L2 pruning. |
| `bayesian_score_pruning.py` | Score-based pruning (BIC, AIC, BDs). |
| `bayesian_csi_pruning.py` | CSI / structural-error pruning. |
| `compare_pruning_methods.py` | Comparison table and progress plotting. |
| `requirements.txt` | Python dependencies. |

## Optional: run only ALARM

If you do not have `bayesian_config.json`, run `python main.py` anyway: the script will run ALARM only and save `alarm_progress.png`.

## ALARM network

The ALARM model is loaded via pgmpy’s `get_example_model('alarm')`. The first time you run, pgmpy may need network access to fetch the model.
