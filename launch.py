"""
launch.py — distribute a config sweep across multiple GPUs.

Each GPU runs one experiment at a time. When it finishes, it pulls the next
job from the shared queue. Already-completed W&B runs are skipped at startup.

Usage:
    python launch.py --config configs/full_sweep.yaml --gpus 0 1 2
    python launch.py --config configs/fraction_ablation.yaml --gpus 0 1
    python launch.py --config configs/quick_test.yaml --gpus 0          # same as training.py
"""

import argparse
import multiprocessing as mp
import os
import time

import wandb
import yaml

from training import expand_config, PROJECT


def worker(gpu_id, queue, existing_run_names):
    """
    Pull experiments from the shared queue and run them one at a time,
    each on the assigned GPU.
    """
    # Set CUDA_VISIBLE_DEVICES so PyTorch Lightning sees only this GPU as device 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import training inside the worker so the environment variable is set first.
    from training import run_sweep

    print(f"[GPU {gpu_id}] Worker started.")

    while True:
        try:
            exp = queue.get(timeout=5)
        except Exception:
            # Queue empty — no more jobs.
            break

        print(f"[GPU {gpu_id}] Starting: {exp}")
        try:
            run_sweep([exp], device=0, existing_run_names=existing_run_names)
        except Exception as e:
            print(f"[GPU {gpu_id}] ERROR on {exp}: {e}")

    print(f"[GPU {gpu_id}] No more jobs — exiting.")


def main():
    parser = argparse.ArgumentParser(description="Launch a sweep across multiple GPUs.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config.")
    parser.add_argument("--gpus",   required=True, type=int, nargs="+",
                        help="GPU indices to use, e.g. --gpus 0 1 2")
    args = parser.parse_args()

    # Expand config into flat experiment list
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    experiments = expand_config(cfg)

    # Fetch already-completed runs once, share across all workers
    print(f"Checking existing W&B runs in '{PROJECT}'...")
    api = wandb.Api()
    try:
        existing_run_names = {run.name for run in api.runs(path=PROJECT)}
    except Exception as e:
        print(f"Error occurred while fetching existing runs: {e}")
        existing_run_names = set()
    pending = [e for e in experiments if _run_name(e) not in existing_run_names]

    print(f"  {len(experiments)} total | {len(experiments) - len(pending)} already done "
          f"| {len(pending)} to run across {len(args.gpus)} GPU(s).\n")

    if not pending:
        print("Nothing to do.")
        return

    # Fill the shared queue
    queue = mp.Queue()
    for exp in pending:
        queue.put(exp)

    # One worker process per GPU
    processes = []
    for gpu_id in args.gpus:
        p = mp.Process(target=worker, args=(gpu_id, queue, existing_run_names))
        p.start()
        processes.append(p)
        time.sleep(2)  # Stagger starts slightly to avoid W&B init collisions

    for p in processes:
        p.join()

    print("\nAll workers finished.")


def _run_name(exp):
    """Reconstruct the run name for a given experiment dict (mirrors training.py logic)."""
    from training import DATASET
    day_label  = "single_day"  if exp.get("single_day",  False) else "multi_day"
    chan_label = "temperature" if exp.get("temperature", False) else "movement"
    frac       = exp.get("train_fraction", 1.0)
    # One name per fold — if any fold is missing we'd still want to run, so
    # we conservatively mark the experiment as pending unless ALL folds exist.
    group = f"{exp['model_type']}_{exp['location']}_{DATASET}_{day_label}_{chan_label}_frac{frac}"
    return group  # checked at fold level inside run_sweep; this is just for the summary count


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
