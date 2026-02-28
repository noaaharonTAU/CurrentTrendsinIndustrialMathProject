# Bayesian Network Pruning Project

Compare pruning methods (Wavelet L2, BIC, AIC, BDs, CSI) on the ALARM benchmark and/or a synthetic Bayesian network. Outputs a comparison table and progress plots.

## Run on Google Colab

1. **Clone the repository** (in a Colab code cell):

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME
```

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your GitHub username and repo name. If the project lives in a subfolder (e.g. `bayesian_pruning_project` is inside the repo), then after cloning use:
`%cd YOUR_REPO_NAME` so that the **parent** of `bayesian_pruning_project` is the current directory.

2. **Install dependencies**:

```python
!pip install -r bayesian_pruning_project/requirements.txt
```

(If the project is at the repo root, use `!pip install -r requirements.txt`.)

3. **Run experiments**:

```python
# ALARM only (no extra files needed)
!python -m bayesian_pruning_project.main --alarm
```

```python
# Synthetic only (requires bayesian_config.json in the project folder)
!python -m bayesian_pruning_project.main --synthetic
```

```python
# Both ALARM and synthetic
!python -m bayesian_pruning_project.main --both
```

If your config file is elsewhere, pass it explicitly:

```python
!python -m bayesian_pruning_project.main --synthetic --config /path/to/bayesian_config.json
```

4. **View results**
   - The **comparison table** is printed in the cell output.
   - **Plots** are saved as `alarm_pruning_progress.png` and/or `synthetic_pruning_progress.png` in the current directory. To show them in the notebook:

```python
from IPython.display import Image, display
display(Image("alarm_pruning_progress.png"))
# display(Image("synthetic_pruning_progress.png"))
```

   - You can also download the images from the Colab file browser (folder icon on the left).

**Tip:** For synthetic experiments, upload `bayesian_config.json` to Colab (e.g. drag-and-drop into the file browser) and run with `--config bayesian_config.json` if you put it in the working directory.

---

## Setup (local)

```bash
cd bayesian_pruning_project
pip install -r requirements.txt
```

For **synthetic** experiments, place `bayesian_config.json` (nodes, edges, cpds, variable_card) in the project root, or set `CONFIG_PATH` in `config.py`.

## Run (local)

From the **repository root** (parent of `bayesian_pruning_project/`):

```bash
# ALARM only
python -m bayesian_pruning_project.main --alarm

# Synthetic only (requires bayesian_config.json)
python -m bayesian_pruning_project.main --synthetic

# Both
python -m bayesian_pruning_project.main --both
```

Or from inside `bayesian_pruning_project/`:

```bash
python main.py --alarm
python main.py --synthetic --config path/to/bayesian_config.json
python main.py --both
```

## Config

Edit `config.py` to tune:

- `min_steps`, `max_steps` — pruning steps
- `CONFIG_PATH` — path to synthetic model JSON
- `ALARM_TARGET_VAR`, `ALARM_INTERVENTIONS` — for alarm metrics
- `SYNTHETIC_TARGET_VAR`, `SYNTHETIC_INTERVENTIONS` — for synthetic
- Data sizes and seeds

## Output

- **Table**: printed comparison (predictive + causal metrics) from the last step of each method.
- **Plot**: progress of metrics vs pruning step, saved as `alarm_pruning_progress.png` and/or `synthetic_pruning_progress.png`.

## Structure

- `config.py` — global parameters
- `helpers.py` — noise, clamp, refit, CPD checks
- `evaluation.py` — log-likelihood, KL, structural error, prediction accuracy, interventional KL, ACE, colliders
- `pruning_wavelet.py` — L2 wavelet pruning
- `pruning_score.py` — BIC/AIC/BDs score-based pruning
- `pruning_structural.py` — CSI/structural-error pruning
- `comparison.py` — build table, print comparison, plot progress
- `data_alarm.py` — load ALARM model and data
- `data_synthetic.py` — load synthetic model and data from JSON
- `main.py` — CLI and run pipeline
