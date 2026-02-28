"""
Global parameters for Bayesian network pruning experiments.
Tune these for alarm, synthetic, or both runs.
"""

# Pruning steps
min_steps = 10
max_steps = 10

# Path to synthetic model config (JSON with nodes, edges, cpds, variable_card)
CONFIG_PATH = "bayesian_config.json"

# Alarm: target variable for prediction accuracy; interventions for interventional KL
ALARM_TARGET_VAR = "CATECHOL"
# List of intervention dicts, e.g. [{"VAR": 0}, {"VAR": 1}]. Empty to skip interventional metrics.
ALARM_INTERVENTIONS = []  # e.g. [{"CATECHOL": 0}, {"CATECHOL": 1}]

# Synthetic: target variable (must exist in synthetic model)
SYNTHETIC_TARGET_VAR = None  # e.g. "dep"
SYNTHETIC_INTERVENTIONS = []

# Data sizes
SYNTHETIC_SAMPLE_SIZE = 4000
SYNTHETIC_TRAIN_RATIO = 0.7
ALARM_TRAIN_SAMPLES = 2000
ALARM_EVAL_SAMPLES = 2000
ALARM_DATA_SAMPLES = 1000  # for fitting initial alarm_model

# Structure learning (synthetic from noisy data)
SYNTHETIC_NOISE_EPS = 0.2
SYNTHETIC_STRUCTURE_SAMPLES = 10  # small for demo; increase for stability

# Random seeds
RANDOM_STATE = 42
