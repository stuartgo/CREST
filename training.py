"""
training.py — run one experiment or a full sweep from a YAML config.

Single run:
    python training.py --location thigh --model_type transformer \
                       --temperature --single_day --train_fraction 0.5

Sweep from config (single GPU):
    python training.py --config configs/full_sweep.yaml

Parallel sweep across multiple GPUs (use launch.py instead):
    python launch.py --config configs/full_sweep.yaml --gpus 0 1 2
"""

import argparse
from itertools import product

import pytorch_lightning as pl
import wandb
import yaml

from data import load_data_kfold
from model import CircadianModel
from params import get_params

DATASET = "hunt"
PROJECT = "CREST-circadian-prediction"


# ── Callbacks ─────────────────────────────────────────────────────────────────

def build_callbacks(group_name, fold):
    return [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath="./checkpoints",
            filename=f"{group_name}-fold{fold}",
            save_top_k=3,
            mode="min",
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        ),
    ]


# ── Single fold ───────────────────────────────────────────────────────────────

def run_fold(model_type, location, temperature, single_day, train_fraction,
             fold, loaders, input_size, device, existing_run_names):
    day_label  = "single_day"  if single_day  else "multi_day"
    chan_label = "temperature" if temperature else "movement"
    group_name = f"{model_type}_{location}_{DATASET}_{day_label}_{chan_label}_frac{train_fraction}"
    run_name   = f"{group_name}_fold{fold}"

    if run_name in existing_run_names:
        print(f"Skipping '{run_name}' — already exists.")
        return

    params = (
        {"learning_rate": 1e-4, "d_model": 1024}
        if model_type == "moment"
        else get_params(model_type, location, temperature, single_day)
    )
    model = CircadianModel(input_size=input_size, output_size=2, params=params, model_type=model_type)

    train_loader, val_loader, test_loader = loaders

    wandb_logger = pl.loggers.WandbLogger(
        project=PROJECT,
        name=run_name,
        tags=[model_type, location, DATASET, day_label, chan_label, f"frac{train_fraction}"],
        group=group_name,
    )
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=[device],
        logger=wandb_logger,
        callbacks=build_callbacks(group_name, fold),
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader)
    wandb_logger.experiment.finish()


# ── Sweep runner ──────────────────────────────────────────────────────────────

def run_sweep(experiments, device, existing_run_names):
    for exp in experiments:
        location       = exp["location"]
        model_type     = exp["model_type"]
        temperature    = exp.get("temperature",    False)
        single_day     = exp.get("single_day",     False)
        train_fraction = exp.get("train_fraction", 1.0)

        print(f"\n{'═' * 60}")
        print(f"  GPU {device} | {model_type} | {location} | temp={temperature} | "
              f"single_day={single_day} | frac={train_fraction}")
        print(f"{'═' * 60}")

        X, y, folds = load_data_kfold(
            location, model_type, single_day, temperature,
            train_fraction=train_fraction,
        )
        input_size = X.shape[1] if model_type == "moment" else X.shape[2]

        for fold_idx, loaders in enumerate(folds, start=1):
            run_fold(
                model_type, location, temperature, single_day,
                train_fraction, fold_idx, loaders, input_size,
                device, existing_run_names,
            )


# ── Config expansion ──────────────────────────────────────────────────────────

def expand_config(cfg):
    """
    Expand a YAML config into a flat list of experiment dicts.
    Any field may be a scalar or a list; all combinations are generated.
    """
    experiments = []
    for block in cfg["experiments"]:
        def as_list(v): return v if isinstance(v, list) else [v]

        for loc, mt, temp, sd, frac in product(
            as_list(block["location"]),
            as_list(block["model_type"]),
            as_list(block.get("temperature",    False)),
            as_list(block.get("single_day",     False)),
            as_list(block.get("train_fraction", 1.0)),
        ):
            experiments.append({
                "location":       loc,
                "model_type":     mt,
                "temperature":    temp,
                "single_day":     sd,
                "train_fraction": frac,
            })
    return experiments


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train circadian acrophase models.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config",   type=str, help="Path to a YAML sweep config file.")
    group.add_argument("--location", type=str, help="Sensor location (thigh | back).")

    parser.add_argument("--model_type",     type=str,  default="transformer",
                        choices=["lstm", "transformer", "tcn", "mamba", "moment"])
    parser.add_argument("--temperature",    action="store_true")
    parser.add_argument("--single_day",     action="store_true")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--device",         type=int,   default=0,
                        help="GPU index to use (default: 0).")
    return parser.parse_args()


def main():
    args = parse_args()

    api = wandb.Api()
    try:
        existing_run_names = {run.name for run in api.runs(path=PROJECT)}
    except Exception as e:
        print(f"Error occurred while fetching existing runs: {e}")
        existing_run_names = set()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        experiments = expand_config(cfg)
        print(f"Loaded {len(experiments)} experiment(s) from '{args.config}'.")
    else:
        experiments = [{
            "location":       args.location,
            "model_type":     args.model_type,
            "temperature":    args.temperature,
            "single_day":     args.single_day,
            "train_fraction": args.train_fraction,
        }]

    run_sweep(experiments, device=args.device, existing_run_names=existing_run_names)


if __name__ == "__main__":
    main()