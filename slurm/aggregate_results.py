#!/usr/bin/env python3
"""
Aggregate the results of a sweep into two CSVs plus a console table.

Reads the text reports written by save_evaluation_report() under
    sweeps/<SWEEP_ID>/<model>/<category>/seed<N>/report/*.txt
and produces:

    runs.csv     one row per run   (model, category, seed, TP/FP/FN/TN, AUROC, ...)
    summary.csv  one row per cell  (model, category) with mean and std over seeds

The sweep grid is pre-specified, so the script declares the expected cells up
front and reports every missing or failed run explicitly instead of quietly
averaging over whatever happens to be on disk.

    python slurm/aggregate_results.py sweeps/sweep_20260915_101500
    python slurm/aggregate_results.py sweeps/my_sweep --seeds 0 1 2 --out-dir .
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

try:
    import yaml
except ImportError:  # provenance becomes best-effort without PyYAML
    yaml = None

MODELS = ["patchcore", "efficientad"]
CATEGORIES = ["carpet", "reda_baseline", "reda_dustOnValidation",
              "reda_dustValidationAndTrain"]

# Metrics parsed out of the report, in the order they appear in the CSV.
METRICS = ["auroc", "aupro", "ap_loc", "f1", "recall", "precision", "accuracy"]
COUNTS = ["tp", "fp", "fn", "tn"]

# save_evaluation_report() writes percentages with two decimals, e.g.
#     Recall      :  98.41% (True defects found)
PATTERNS = {
    "accuracy":        r"Accuracy\s*:\s*([\d.]+)%",
    "precision":       r"Precision\s*:\s*([\d.]+)%",
    "recall":          r"Recall\s*:\s*([\d.]+)%",
    "f1":              r"F1-Score\s*:\s*([\d.]+)%",
    "auroc":           r"AUROC\s*:\s*([\d.]+)%",
    "aupro":           r"AUPRO\s*:\s*([\d.]+)%",
    "ap_loc":          r"AP-loc\s*:\s*([\d.]+)%",
    "image_threshold": r"Image threshold\s*:\s*([\d.]+)",
    "pixel_threshold": r"Pixel threshold\s*:\s*([\d.]+)",
    "tn":              r"TN:\s*(\d+)",
    "fp":              r"FP:\s*(\d+)",
    "fn":              r"FN:\s*(\d+)",
    "tp":              r"TP:\s*(\d+)",
}


@dataclass
class Run:
    model: str
    category: str
    seed: int
    status: str = "ok"
    timestamp: str = ""
    backbone: str = ""
    layers: str = ""
    num_epochs: str = ""
    train_batch_size: str = ""
    dataset_version: str = ""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    accuracy: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")
    auroc: float = float("nan")
    aupro: float = float("nan")
    ap_loc: float = float("nan")
    image_threshold: float = float("nan")
    pixel_threshold: float = float("nan")
    report_file: str = ""


def parse_report(text: str) -> dict:
    """Extract every known field from a report. Missing fields are simply absent."""
    out: dict = {}
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, text)
        if m is None:
            continue
        out[key] = int(m.group(1)) if key in COUNTS else float(m.group(1))
    return out


def read_provenance(run_dir: Path) -> dict:
    """Pull backbone / layers / epochs from the config copy saved by save_config_file()."""
    if yaml is None:
        return {}
    config_dir = run_dir / "config"
    candidates = sorted(config_dir.glob("*.yaml")) if config_dir.is_dir() else []
    if not candidates:
        return {}
    try:
        cfg = yaml.safe_load(candidates[-1].read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    arch = cfg.get("model_architecture", {})
    layers = arch.get("layers", [])
    model_key = cfg.get("run", {}).get("model", "")
    model_cfg = cfg.get(f"{model_key}_configuration", {})

    return {
        "backbone": str(arch.get("backbone", "")),
        "layers": "_".join(layers) if isinstance(layers, list) else str(layers),
        "num_epochs": str(model_cfg.get("num_epochs", "")),
        "train_batch_size": str(model_cfg.get("train_batch_size", "")),
        "dataset_version": str(cfg.get("dataset_pipeline", {}).get("dataset_version", "")),
    }


def collect_run(sweep_dir: Path, model: str, category: str, seed: int) -> Run:
    run = Run(model=model, category=category, seed=seed)
    run_dir = sweep_dir / model / category / f"seed{seed}"

    if not run_dir.is_dir():
        run.status = "not_launched"
        return run

    reports = sorted((run_dir / "report").glob("*evaluation_report*.txt")) \
        if (run_dir / "report").is_dir() else []
    if not reports:
        # Directory exists but no report: the job died before the metric block.
        run.status = "failed" if (run_dir / "train.log").exists() else "no_report"
        return run

    if len(reports) > 1:
        print(f"[warn] {len(reports)} reports in {run_dir / 'report'}, "
              f"using the most recent ({reports[-1].name})", file=sys.stderr)

    report = reports[-1]
    parsed = parse_report(report.read_text(encoding="utf-8", errors="replace"))
    if "auroc" not in parsed:
        run.status = "unparseable"
        run.report_file = str(report)
        return run

    for key, value in parsed.items():
        setattr(run, key, value)
    for key, value in read_provenance(run_dir).items():
        setattr(run, key, value)

    run.timestamp = report.name.split("_evaluation_report")[0]
    run.report_file = str(report)
    return run


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def build_summary(runs: list[Run], seeds: list[int],
                  models: list[str], categories: list[str]) -> list[dict]:
    rows = []
    for model in models:
        for category in categories:
            cell = [r for r in runs if r.model == model and r.category == category]
            ok = [r for r in cell if r.status == "ok"]

            row = {
                "model": model,
                "category": category,
                "n_seeds_ok": len(ok),
                "n_seeds_expected": len(seeds),
                "seeds_ok": "|".join(str(r.seed) for r in ok),
                "seeds_missing": "|".join(
                    f"{r.seed}:{r.status}" for r in cell if r.status != "ok"
                ),
            }
            for metric in METRICS:
                m, s = mean_std([getattr(r, metric) for r in ok])
                row[f"{metric}_mean"] = round(m, 4) if m == m else ""
                row[f"{metric}_std"] = round(s, 4) if s == s else ""
            for count in COUNTS:
                m, _ = mean_std([float(getattr(r, count)) for r in ok])
                row[f"{count}_mean"] = round(m, 2) if m == m else ""
            rows.append(row)
    return rows


def print_table(summary: list[dict]) -> None:
    header = f"{'model':<13} {'category':<26} {'n':>3}  {'AUROC':>16} {'F1':>16} {'Recall':>16}"
    print("\n" + header)
    print("-" * len(header))
    for row in summary:
        def cell(metric: str) -> str:
            mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
            return "--".rjust(16) if mean == "" else f"{mean:6.2f} +/- {std:5.2f}"
        print(f"{row['model']:<13} {row['category']:<26} "
              f"{row['n_seeds_ok']:>3}  {cell('auroc')} {cell('f1')} {cell('recall')}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path, help="sweeps/<SWEEP_ID>")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 42, 101])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write runs.csv and summary.csv (default: the sweep dir)")
    args = ap.parse_args()

    sweep_dir: Path = args.sweep_dir
    if not sweep_dir.is_dir():
        print(f"[error] sweep directory not found: {sweep_dir}", file=sys.stderr)
        return 2
    out_dir = args.out_dir or sweep_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        collect_run(sweep_dir, model, category, seed)
        for model in args.models
        for category in args.categories
        for seed in args.seeds
    ]

    runs_csv = out_dir / "runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(Run)])
        writer.writeheader()
        for run in runs:
            writer.writerow(asdict(run))

    summary = build_summary(runs, args.seeds, args.models, args.categories)
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print_table(summary)

    n_ok = sum(1 for r in runs if r.status == "ok")
    print(f"{n_ok}/{len(runs)} runs aggregated")
    incomplete = [r for r in runs if r.status != "ok"]
    if incomplete:
        print(f"\n{len(incomplete)} run(s) missing from the grid:")
        for r in incomplete:
            print(f"  - {r.model:<12} {r.category:<26} seed {r.seed}  [{r.status}]")
    print(f"\nwrote {runs_csv}\nwrote {summary_csv}")

    return 0 if n_ok == len(runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
