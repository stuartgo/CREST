import argparse

import optuna
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback

from data import load_data_split
from model import CircadianModel

DATASET = "hunt"
PROJECT = "circ_prediction_hyperopt2"


# ── Hyperparameter search spaces ──────────────────────────────────────────────

def suggest_params(trial, model_name):
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)

    if model_name == "transformer":
        return {
            "d_model":         trial.suggest_categorical("d_model", [64, 128, 256, 512]),
            "num_heads":       trial.suggest_categorical("num_heads", [2, 4, 8]),
            "num_layers":      trial.suggest_int("num_layers", 2, 6),
            "dim_feedforward": trial.suggest_int("dim_feedforward", 128, 256),
            "dropout":         trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate":   lr,
        }

    if model_name == "mamba":
        return {
            "d_model":       trial.suggest_categorical("d_model", [64, 128, 256, 512]),
            "d_state":       trial.suggest_categorical("d_state", [16, 32, 64]),
            "d_conv":        trial.suggest_categorical("d_conv", [2, 3, 4]),
            "expand":        trial.suggest_categorical("expand", [2, 4]),
            "learning_rate": lr,
        }

    if model_name == "tcn":
        num_channels = trial.suggest_categorical(
            "num_channels",
            [[64, 64, 64], [128, 128, 128], [256, 256, 256], [512, 512, 512]],
        )
        return {
            "d_model":       num_channels[0],
            "num_channels":  num_channels,
            "kernel_size":   trial.suggest_int("kernel_size", 2, 8),
            "dropout":       trial.suggest_float("dropout", 0.0, 0.5),
            "activation":    trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        }

    if model_name == "lstm":
        return {
            "d_model":       trial.suggest_int("d_model", 64, 512),
            "num_layers":    trial.suggest_int("num_layers", 1, 4),
            "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
            "dropout":       trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        }

    raise ValueError(f"Unknown model name: {model_name}")


# ── Objective ─────────────────────────────────────────────────────────────────

def objective(trial, model_name, location, temperature, single_day):
    params = suggest_params(trial, model_name)

    train_loader, val_loader, _ = load_data_split(location, temperature, single_day)

    model = CircadianModel(input_size=3, output_size=2, model_type=model_name, params=params)

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        ],
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics["val_loss"].item()


# ── Study ─────────────────────────────────────────────────────────────────────

def run_optimization(model_name, location, temperature, single_day, n_trials=50):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=15)
    study  = optuna.create_study(direction="minimize", pruner=pruner)

    group = f"{model_name}_{location}_{DATASET}_{temperature}_{single_day}"
    wandbc = WeightsAndBiasesCallback(
        wandb_kwargs={
            "project": PROJECT,
            "tags": [model_name, location, f"single_day_{single_day}", f"temperature_{temperature}"],
            "group": group,
        },
        as_multirun=True,
    )
    study.optimize(
        lambda trial: objective(trial, model_name, location, temperature, single_day),
        n_trials=n_trials,
        callbacks=[wandbc],
    )
    return study


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimisation")
    parser.add_argument("--location",     required=True)
    parser.add_argument("--temperature",  action="store_true")
    parser.add_argument("--single_day",   action="store_true")
    args = parser.parse_args()

    for model_type in ["mamba", "lstm", "transformer", "tcn"]:
        run_optimization(
            model_name=model_type,
            location=args.location,
            temperature=args.temperature,
            single_day=args.single_day,
        )


if __name__ == "__main__":
    main()
