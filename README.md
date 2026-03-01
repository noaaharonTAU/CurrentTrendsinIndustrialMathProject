# Bayesian Network Pruning Project

Compare pruning methods (Wavelet L2, BIC, AIC, BDs, CSI) on the ALARM benchmark and/or a synthetic Bayesian network. Outputs a comparison table and progress plots.

## Run on Google Colab

1. **Clone the repository** (in a Colab code cell):

```python
!git clone https://github.com/noaaharonTAU/CurrentTrendsinIndustrialMathProject.git
%cd CurrentTrendsinIndustrialMathProject
```

2. **Install dependencies**:

```python
!pip install -r requirements.txt
```

3. **Run main**:

```python
# ALARM only 
!python main.py --alarm
```

```python
# Synthetic only 
!python main.py --synthetic
```

```python
# Both ALARM and synthetic
!python main.py --both
```

Run **5 times** and report **averaged** comparison table and plots:

```python
!python main.py --alarm --runs 5
# or
!python main.py --both --runs 5
```

4. **View results**
   - The **comparison table** is printed in the cell output.
   - **Plots** are saved as `alarm_pruning_progress.png` and/or `synthetic_pruning_progress.png`. To show them in the notebook:

```python
from IPython.display import Image, display
display(Image("alarm_pruning_progress.png"))
# display(Image("synthetic_pruning_progress.png"))
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
