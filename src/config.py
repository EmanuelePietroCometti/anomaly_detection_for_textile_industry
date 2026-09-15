import os
import yaml
from pathlib import Path

# Nomi delle classi anomalib, servono a costruire il path del symlink.
MODEL_CLASS = {
    "patchcore": "Patchcore",
    "efficientad": "EfficientAd",
    "rd4ad": "ReverseDistillation",
    "supersimplenet": "SuperSimpleNet",
}

def load_config(config_path=None):
    """
    Resolution order:
      1. explicit `config_path` argument
      2. the AD_CONFIG environment variable (used by the SLURM sweep)
      3. the repository default, "config.yaml"
    """
    if config_path is None:
        config_path = os.environ.get("AD_CONFIG", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def apply_run_dir(config: dict, run_dir: str, baseline: str) -> dict:
    """Riscrive ogni path di output sotto run_dir.

    È l'unica cosa che impedisce a 40 run di sovrascriversi report, checkpoint
    ed export a vicenda. Muta config in place e lo restituisce.
    """
    run_dir = Path(run_dir)
    model_class = MODEL_CLASS.get(baseline, baseline)
    category = config.get("datamodule_configuration", {}).get("category", "dataset")

    paths = config.setdefault("paths", {})
    paths["default_root_dir"] = str(run_dir / "results")
    paths["symlink_path"] = str(run_dir / "results" / model_class / category / "latest")
    paths["anomaly_images"] = str(run_dir / "anomaly_images")
    paths["report_path"] = str(run_dir / "report")
    paths["auroc_path"] = str(run_dir / "AUROC")
    paths["eda_path"] = str(run_dir / "EDA")
    paths["checkpoint_dir"] = str(run_dir / "checkpoints")
    paths["checkpoint_destination"] = str(run_dir / "checkpoints" / f"{model_class}-tested.ckpt")
    paths["exports_pt_path"] = str(run_dir / "exports")
    paths["config_dst_path"] = str(run_dir / "config")

    run_dir.mkdir(parents=True, exist_ok=True)
    return config