# Bayesian Network Pruning Project

Compare pruning methods (Wavelet L2, BIC, AIC, BDs, CSI) on the ALARM benchmark and/or a synthetic Bayesian network. Outputs a comparison table and progress plots.

## Requirements

- **Python 3.8 or 3.9+** (tested with 3.9 and 3.10). The dependencies in `requirements.txt` are not compatible with Python 3.7 or below.
- Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `pgmpy`, `networkx`, `matplotlib` (see `requirements.txt`).

---

## Option A: Run locally (terminal or PyCharm)

Use this if you already have Python 3.8+ on your machine.

### 1. Clone the repository

In a terminal:

```bash
git clone https://github.com/noaaharonTAU/CurrentTrendsinIndustrialMathProject.git
cd CurrentTrendsinIndustrialMathProject
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you prefer not to use a venv, run `pip install -r requirements.txt` from the project folder (use `python3` and `pip3` if your system defaults to Python 2).

### 3. Run from the terminal

From the project folder (where `main.py` is):

```bash
# ALARM only (no extra files needed)
python main.py --alarm

# Synthetic only (requires bayesian_config.json in the project folder)
python main.py --synthetic

# Both
python main.py --both

# Run 5 times and average results (parallel workers)
python main.py --alarm --runs 5
```

### 4. Run from PyCharm

1. **File → Open** and select the `CurrentTrendsinIndustrialMathProject` folder.
2. Set the Python interpreter: **File → Settings → Project → Python Interpreter** → Add → Existing (point to your Python 3.8+ or to the project’s `venv/bin/python`).
3. Install dependencies: open the **Terminal** tab at the bottom and run `pip install -r requirements.txt`.
4. Right‑click `main.py` → **Run 'main'**, or add **Run → Edit Configurations** and set script path to `main.py` and parameters to e.g. `--alarm` or `--both`.
5. Alternatively, run the same commands as above in the PyCharm terminal: `python main.py --alarm`, etc.

For **synthetic** runs, ensure `bayesian_config.json` is in the project folder (or set `CONFIG_PATH` in `config.py`).

---

## Option B: Run on Google Colab

Use this if you don’t have Python 3.8+ locally or prefer to run in the cloud. Colab provides a compatible Python environment.

### 1. Clone and go to the project folder

**In a Colab code cell:**

```python
!git clone https://github.com/noaaharonTAU/CurrentTrendsinIndustrialMathProject.git
%cd CurrentTrendsinIndustrialMathProject
```

**Or in the Colab terminal** (Tools → Terminal, or the terminal icon):

```bash
git clone https://github.com/noaaharonTAU/CurrentTrendsinIndustrialMathProject.git
cd CurrentTrendsinIndustrialMathProject
```

### 2. Install dependencies

**In a Colab code cell:**

```python
!pip install -r requirements.txt
```

**Or in the Colab terminal:**

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

**In a Colab code cell:**

```python
# ALARM only (no extra files needed)
!python main.py --alarm
```

```python
# Synthetic only (requires bayesian_config.json in the project folder)
!python main.py --synthetic
```

```python
# Both ALARM and synthetic
!python main.py --both
```

```python
# Run 5 times and average (parallel workers; total time ~one run)
!python main.py --alarm --runs 5
# or
!python main.py --both --runs 5
```

**Or in the Colab terminal** (after `cd CurrentTrendsinIndustrialMathProject` and `pip install -r requirements.txt`):

```bash
python main.py --alarm
# or: python main.py --synthetic   or   python main.py --both
```

### 4. View results

- The **comparison table** is printed in the cell output (or terminal output).
- **Plots** are saved as `alarm_pruning_progress.png` and/or `synthetic_pruning_progress.png`. To show them in a notebook cell:

```python
from IPython.display import Image, display
display(Image("alarm_pruning_progress.png"))
# display(Image("synthetic_pruning_progress.png"))
```

You can also open the **Files** panel (folder icon) and download the PNGs.

**Tip:** For synthetic runs, upload `bayesian_config.json` to Colab (e.g. drag-and-drop into the file browser) and run with `--config bayesian_config.json` if it’s in the current directory.

---

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
- **Histories** (optional): with `--save-histories PATH`, per-step metrics for each method are written to a JSON file so you can load and inspect or re-plot them later.

To save histories (local or Colab):

```bash
python main.py --alarm --save-histories alarm_histories.json
python main.py --both --save-histories hist   # writes hist_alarm.json and hist_synthetic.json
```

With `--runs N`, the saved file contains `runs` (one history per run) and `aggregated` (mean per step).

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
