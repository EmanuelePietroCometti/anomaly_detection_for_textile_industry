#!/usr/bin/env python3
"""
Generate a per-run config.yaml derived from the repository template.

Every run of the sweep (model x category x seed) gets its own YAML with:
  - datamodule_configuration.category  -> the MVTec class under test
  - run.seed                           -> consumed by main.py --seed / seed_everything
  - efficientad_configuration.imagenette_dir -> the staged copy on $SCRATCH_FLASH
  - run.gpu_slot                       -> which GPU of the node this run used
  - paths.*                            -> all rewritten under the run directory, so
                                          the 4 runs packed on one node (and every
                                          other job) never overwrite each other's
                                          results, checkpoints or exports.

With --check-only the pre-flight validation runs and nothing is written: the
staging job uses it to reject a broken grid before a GPU node is allocated.

The generated file is also the artefact copied by save_config_file(), so each run
ships a complete, self-describing record of the configuration it was trained with.
"""

import argparse
import sys
from pathlib import Path

import yaml

# Anomalib class names, used to build result paths that match the Engine layout.
MODEL_CLASS = {
    "patchcore": "Patchcore",
    "efficientad": "EfficientAd",
}


def check_dataset(dataset_root: Path, category: str, model: str,
                  imagenette_dir: Path) -> list[str]:
    """Pre-flight validation. Returns a list of fatal problems (empty == OK)."""
    problems: list[str] = []
    cat_dir = dataset_root / category

    if not cat_dir.is_dir():
        problems.append(f"missing category directory: {cat_dir}")
        return problems

    for sub in ("train/good", "test/good"):
        if not (cat_dir / sub).is_dir():
            problems.append(f"missing required directory: {cat_dir / sub}")

    test_dir = cat_dir / "test"
    if test_dir.is_dir():
        abnormal = [d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]
        if not abnormal:
            problems.append(f"no abnormal subfolder under {test_dir} (test/good only)")

    # run_anomaly_pipeline() passes mask_dir="ground_truth" unconditionally and the
    # metric block concatenates pixel masks; without them the run dies after training.
    if not (cat_dir / "ground_truth").is_dir():
        problems.append(
            f"missing {cat_dir / 'ground_truth'} - the pixel-level metric block in "
            "run_anomaly_pipeline() needs it (use --skip-dataset-check to bypass)"
        )

    if model == "efficientad" and not imagenette_dir.is_dir():
        problems.append(f"EfficientAD needs the imagenette directory: {imagenette_dir}")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default="config.yaml", help="repo config.yaml used as base")
    ap.add_argument("--model", required=True, choices=sorted(MODEL_CLASS))
    ap.add_argument("--category", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset-root", default="./data/mvtec",
                    help="directory holding the categories (on Legion: the copy "
                         "staged under $SCRATCH_FLASH)")
    ap.add_argument("--imagenette-dir", default=None,
                    help="override efficientad_configuration.imagenette_dir")
    ap.add_argument("--slurm-job-id", default="",
                    help="recorded under run.slurm_job_id for traceability")
    ap.add_argument("--gpu-slot", default="",
                    help="GPU index this run was pinned to (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--check-only", action="store_true",
                    help="run the dataset pre-flight and exit without writing a config")
    ap.add_argument("--run-dir", help="output directory for this single run")
    ap.add_argument("--out", help="path of the generated config.yaml")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="override datamodule num_workers (pass $SLURM_CPUS_PER_TASK)")
    ap.add_argument("--skip-dataset-check", action="store_true")
    args = ap.parse_args()

    template = Path(args.template)
    if not template.is_file():
        print(f"[make_run_config] ERROR: template not found: {template}", file=sys.stderr)
        return 2

    if not args.check_only and (args.run_dir is None or args.out is None):
        print("[make_run_config] ERROR: --run-dir and --out are required "
              "unless --check-only is given", file=sys.stderr)
        return 2

    cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
    model_class = MODEL_CLASS[args.model]

    ead = cfg.setdefault("efficientad_configuration", {})
    if args.imagenette_dir is not None:
        ead["imagenette_dir"] = args.imagenette_dir
    imagenette_dir = Path(ead.get("imagenette_dir", "./data/imagenette_for_efficientad"))

    if not args.skip_dataset_check:
        problems = check_dataset(Path(args.dataset_root), args.category,
                                 args.model, imagenette_dir)
        if problems:
            print(f"[make_run_config] dataset pre-flight FAILED for "
                  f"{args.model}/{args.category}:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 3

    if args.check_only:
        print(f"[make_run_config] pre-flight OK: {args.model} | {args.category}")
        return 0

    run_dir = Path(args.run_dir)

    # --- run provenance -------------------------------------------------------
    cfg["run"] = {
        "model": args.model,
        "category": args.category,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "slurm_job_id": args.slurm_job_id,
        "gpu_slot": args.gpu_slot,
    }

    # --- dataset selection ----------------------------------------------------
    dm = cfg.setdefault("datamodule_configuration", {})
    dm["root"] = args.dataset_root
    dm["category"] = args.category
    if args.num_workers is not None:
        dm["num_workers"] = args.num_workers

    # Keep the split seed aligned, in case --create-dataset is ever used here.
    cfg.setdefault("dataset_pipeline", {})["seed"] = args.seed

    # --- output isolation -----------------------------------------------------
    paths = cfg.setdefault("paths", {})
    paths["default_root_dir"] = str(run_dir / "results")
    paths["symlink_path"] = str(run_dir / "results" / model_class / args.category / "latest")
    paths["anomaly_images"] = str(run_dir / "anomaly_images")
    paths["report_path"] = str(run_dir / "report")
    paths["auroc_path"] = str(run_dir / "AUROC")
    paths["eda_path"] = str(run_dir / "EDA")
    paths["checkpoint_dir"] = str(run_dir / "checkpoints")
    paths["checkpoint_destination"] = str(run_dir / "checkpoints" / f"{model_class}-tested.ckpt")
    paths["exports_pt_path"] = str(run_dir / "exports")
    paths["config_src_path"] = str(Path(args.out).resolve())
    paths["config_dst_path"] = str(run_dir / "config")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False,
                                  allow_unicode=True), encoding="utf-8")

    print(f"[make_run_config] {args.model} | {args.category} | seed={args.seed} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
