# Circadian Acrophase Prediction

Predicts the **circadian acrophase** (the peak time of the activity/temperature rhythm) from wrist/thigh actigraphy recordings using deep sequence models. Part of the HUNT4 study.

## Overview

The acrophase is encoded as a circular quantity — a (sin, cos) pair — to avoid the discontinuity at midnight. Models are trained to minimise MSE on this representation; evaluation uses **circular MAE in hours**.

### Supported models

| Model | Key |
|---|---|
| LSTM | `lstm` |
| Transformer (custom encoder) | `transformer` |
| Temporal Convolutional Network | `tcn` |
| Mamba2 (state-space model) | `mamba` |
| MOMENT (foundation model embeddings) | `moment` |

---

## File structure

```
├── model.py              # CircadianModel — PyTorch Lightning module, all architectures
├── TransformerEncoder.py # Custom multi-head attention transformer encoder
├── data.py               # Data loading, preprocessing, and k-fold / split helpers
├── params.py             # Best hyperparameters from Optuna sweeps (PARAMS dict)
├── training.py           # Final training loop with k-fold CV, W&B logging
├── hyperopt.py           # Optuna hyperparameter search
└── example.py            # Self-contained demo with synthetic data (no real data needed)
```

---

## Input features

Each timestep has 3 input channels:

1. **Activity** (raw accelerometry count) *or* **skin temperature** depending on the `--temperature` flag
2. **sin(hour-of-day)** — circular time encoding
3. **cos(hour-of-day)** — circular time encoding

Time encoding uses the recording start time so the model knows where in the 24-hour cycle each window begins.

---

## Data format

Real data is loaded from `.pt` files under `/data/stuartgo/hunt4/temp_act/`:

| File | Contents |
|---|---|
| `X_{location}.pt` | `(N, T, C)` raw sensor tensor |
| `y_{location}.pt` | `(N, cols)` target tensor; column 1 is acrophase in hours |
| `other_info_{location}.pt` | `(N, cols)` metadata; column 0 is subject ID, column 2 is recording start hour |
| `embeddings_{location}_{single_day}_{temperature}_moment.pt` | Pre-computed MOMENT embeddings |

`location` is `thigh` or `back`.

---

## Running

### Hyperparameter optimisation

```bash
python hyperopt.py --location thigh
python hyperopt.py --location thigh --temperature
python hyperopt.py --location thigh --single_day
python hyperopt.py --location back --temperature --single_day
```

Results are logged to the `circ_prediction_hyperopt2` W&B project. Best parameters should be added to `params.py`.

### Final training

Edit the loop variables at the bottom of `training.py` to select the configurations you want to run, then:

```bash
python training.py
```

Runs 5-fold cross-validation for each combination of location / model / temperature / single_day / train_fraction. Results are logged to the `circ_prediction_final` W&B project. Already-completed runs are detected via the W&B API and skipped automatically.

### Demo (no real data required)

```bash
python example.py
```

Generates synthetic actigraphy data and trains LSTM, Transformer, and TCN for a few epochs, printing test circular MAE for each.

---

## Data splits

Splits are always **subject-wise** to prevent data leakage between splits — all recordings from a given subject stay in the same partition.

**Hyperopt** uses a single 60/20/20 subject split (train/val/test).

**Training** uses 5-fold CV: each fold assigns 80% of non-test subjects to train and 20% to val. Val and test sizes are always the full complement; only the training set is affected by `train_fraction`.

### Note on hyperopt–training leakage

Hyperopt and k-fold training use independently shuffled subject orderings, so there is no strict guarantee that hyperopt's val/test subjects fall outside training's train folds. This is a known and accepted limitation of the hyperopt → fixed-params → k-fold CV pipeline. The leakage is indirect (only aggregate val loss influences hyperparameter selection) and nested CV would be prohibitively expensive. Final reported metrics come from the k-fold test folds, which are never seen during either hyperopt or training.

---

## Training fraction

`train_fraction` controls what fraction of **training subjects** (not samples) are used per fold, allowing data-efficiency experiments. Val and test folds are always kept at full size. Example values in `training.py`: `[0.1, 0.2, 0.3, 0.4, 0.5, 1.0]`.

---

## Dependencies

```
torch
pytorch_lightning
numpy
scikit-learn
wandb
optuna
pytorch_tcn
mamba_ssm       # for mamba model
momentfm        # for moment model (optional)
```
