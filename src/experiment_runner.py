"""
experiment_runner.py — Runs all 120 experiments

EXPERIMENT FLOW (single run):
──────────────────────────────
1. Sample n images from ImageNet val set (calibration set)
2. Load pretrained FP32 model (no training required)
3. Run calibration images through model → measure activation ranges
4. Convert FP32 → INT8
5. Evaluate INT8 model on val set
6. Measure accuracy drop, ECE, model size, latency
7. Save result as JSON

TOTAL: 4 models × 6 calibration sizes × 5 seeds = 120 runs

Usage:
  python src/experiment_runner.py                    # run all
  python src/experiment_runner.py --resume           # continue from where left off
  python src/experiment_runner.py --model resnet18 --size 100 --seed 42
"""

import os
import sys
import json
import argparse
import traceback

import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from config import (
    MODELS, CALIB_SIZES, SEEDS, DEVICE,
    RESULTS_RAW_DIR, RESULTS_AGG_DIR,
)
from src.data_utils import get_val_loader, get_calibration_subset
from src.model_utils import load_model, evaluate_model
from src.quantization import full_ptq_pipeline
from src.metrics import compute_all_metrics


def _result_path(model_name, calib_size, seed):
    os.makedirs(RESULTS_RAW_DIR, exist_ok=True)
    return os.path.join(RESULTS_RAW_DIR, f"{model_name}_{calib_size}_{seed}.json")


def _is_done(model_name, calib_size, seed):
    return os.path.exists(_result_path(model_name, calib_size, seed))


def _save(result):
    path = _result_path(result["model"], result["calib_size"], result["seed"])
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def run_single_experiment(model_name: str, calib_size: int, seed: int) -> dict:
    """
    Runs a single experiment:
      model_name : which model (resnet18, resnet50, ...)
      calib_size : number of calibration images (10, 50, 100, ...)
      seed       : random seed for reproducibility
    """
    print(f"\n[RUN] {model_name} | n={calib_size} | seed={seed}")

    # Validation loader: used to evaluate both FP32 and INT8 accuracy
    val_loader   = get_val_loader(batch_size=64, max_samples=5000)

    # Calibration loader: used ONLY to measure activation ranges (not for training)
    calib_loader = get_calibration_subset(calib_size, seed)

    # Load pretrained FP32 model (ImageNet weights, no training needed)
    fp32_model = load_model(model_name, device=DEVICE)
    fp32_acc   = evaluate_model(fp32_model, val_loader, DEVICE)
    print(f"  FP32 accuracy: {fp32_acc:.2f}%")

    # PTQ: convert FP32 → INT8 using calibration set
    int8_model = full_ptq_pipeline(fp32_model, model_name, calib_loader, DEVICE)

    # Compute all metrics
    metrics = compute_all_metrics(
        fp32_model, int8_model, val_loader,
        device=DEVICE, fp32_acc=fp32_acc,
    )

    result = {"model": model_name, "calib_size": calib_size, "seed": seed, **metrics}
    _save(result)

    print(f"  INT8 accuracy: {metrics['int8_accuracy']:.2f}% | "
          f"ΔAcc: {metrics['accuracy_drop']:.4f}% | "
          f"ECE: {metrics['ece']:.4f} | "
          f"Size: {metrics['model_size_mb']:.1f}MB | "
          f"Latency: {metrics['latency_ms']:.1f}ms")
    return result


def run_all_experiments(resume: bool = False, models: list = None):
    """
    Runs experiments for the given models (default: all 4 models).
    resume=True → skips completed runs, continues from where left off.
    models=['resnet18'] → run only that model (30 runs instead of 120).
    """
    model_list = models if models is not None else MODELS
    combos = [(m, s, seed) for m in model_list for s in CALIB_SIZES for seed in SEEDS]

    print(f"Models: {model_list}")
    print(f"Total runs planned: {len(combos)}")

    if resume:
        todo = [(m, s, seed) for m, s, seed in combos if not _is_done(m, s, seed)]
        print(f"Resume mode: {len(combos)-len(todo)} completed, {len(todo)} remaining.")
    else:
        todo = combos

    results, failed = [], []

    for model_name, calib_size, seed in tqdm(todo, desc="Experiments"):
        try:
            r = run_single_experiment(model_name, calib_size, seed)
            results.append(r)
        except Exception as e:
            print(f"\n  [ERROR] {model_name} n={calib_size} seed={seed}: {e}")
            traceback.print_exc()
            failed.append({"model": model_name, "calib_size": calib_size,
                           "seed": seed, "error": str(e)})

    print(f"\n✓ Completed: {len(results)} | ✗ Failed: {len(failed)}")
    return results


def aggregate_results() -> pd.DataFrame:
    """
    Reads all JSON results, computes mean ± std per (model, calib_size),
    and saves to CSV.
    """
    os.makedirs(RESULTS_AGG_DIR, exist_ok=True)
    records = []
    for fname in os.listdir(RESULTS_RAW_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(RESULTS_RAW_DIR, fname)) as f:
                records.append(json.load(f))

    if not records:
        print("No results found yet.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS_AGG_DIR, "all_results.csv"), index=False)

    metric_cols = ["fp32_accuracy", "int8_accuracy", "accuracy_drop",
                   "ece", "model_size_mb", "latency_ms"]
    rows = []
    for (model_name, size), grp in df.groupby(["model", "calib_size"]):
        row = {"model": model_name, "calib_size": size}
        for col in metric_cols:
            row[f"{col}_mean"] = round(grp[col].mean(), 4)
            row[f"{col}_std"]  = round(grp[col].std(), 4)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["model", "calib_size"])
    summary.to_csv(os.path.join(RESULTS_AGG_DIR, "summary.csv"), index=False)
    print(f"Summary saved: {RESULTS_AGG_DIR}/summary.csv")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=str)
    parser.add_argument("--size",      type=int)
    parser.add_argument("--seed",      type=int)
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        aggregate_results()
    elif args.model and args.size and args.seed:
        run_single_experiment(args.model, args.size, args.seed)
        aggregate_results()
    else:
        run_all_experiments(resume=args.resume)
        aggregate_results()
