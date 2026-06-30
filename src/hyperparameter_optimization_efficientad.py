"""
Hyperparameter Optimization (HPO) for EfficientAD using Optuna.
Saves trials to a local SQLite database to enable real-time tracking 
via optuna-dashboard.
"""

import os
import gc
import torch
import optuna
import tempfile
from pathlib import Path

from anomalib.data import Folder
from anomalib.data.utils.split import ValSplitMode
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.lightning_model import EfficientAdModelSize

from config import load_config

def objective(trial, base_config):
    """
    Objective function for Optuna. Evaluates EfficientAD performance
    by exploring model sizes, learning rates, and weight decay.
    """
    # Define the Search Space specifically for EfficientAD
    model_size_choice = trial.suggest_categorical("model_size", ["s", "m"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    enum_size = EfficientAdModelSize.S if model_size_choice == "s" else EfficientAdModelSize.M

    # Setup DataModule
    datamodule_cfg = base_config.get("datamodule_configuration", {})
    root_path = Path(datamodule_cfg.get("root", "./data/mvtec"))
    category = datamodule_cfg.get("category", "reda")
    dataset_path = root_path / category
    
    test_path = dataset_path / "test"
    abnormal_dirs = [f"test/{d}" for d in os.listdir(test_path) if (test_path / d).is_dir() and d != "good"] if test_path.exists() else []

    # EfficientAD strictly requires train_batch_size=1 due to its internal student-teacher penalty logic
    datamodule = Folder(
        name=category,
        root=str(dataset_path),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir=abnormal_dirs,
        mask_dir="ground_truth",
        train_batch_size=1, 
        eval_batch_size=base_config.get("efficientad_configuration", {}).get("eval_batch_size", 1),
        num_workers=datamodule_cfg.get("num_workers", 4),
        val_split_mode=ValSplitMode.SAME_AS_TEST
    )

    # Model Initialization
    imagenette_dir = base_config.get("efficientad_configuration", {}).get("imagenette_dir", "./data/imagenette_for_efficientad")
    
    model = EfficientAd(
        imagenet_dir=imagenette_dir,
        model_size=enum_size,
        lr=learning_rate,
        weight_decay=weight_decay,
        padding=True,
        pad_maps=False
    )

    # Create a temporary directory for local checkpoints and logs to avoid clutter
    with tempfile.TemporaryDirectory() as tmp_dir:
        
        # Engine Setup
        # Fixed to 30 max_epochs as requested
        engine = Engine(
            max_epochs=30,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=base_config.get("general_configuration", {}).get("precision", "32-true"),
            default_root_dir=tmp_dir,
            logger=False
        )

        # Training
        engine.fit(model=model, datamodule=datamodule)

        # Evaluation
        test_results = engine.test(model=model, datamodule=datamodule)

    # Explicit memory cleanup to prevent GPU VRAM saturation over multiple trials
    del engine
    del model
    del datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Metric Extraction & Fitness Calculation
    if not test_results or len(test_results) == 0:
        raise ValueError("No metrics returned by engine.test()")
        
    metrics = test_results[0]
    image_f1 = metrics.get("image_F1Score", 0.0)
    pixel_f1 = metrics.get("pixel_F1Score", 0.0)
    
    # Composite Score: Balanced equally between global detection (image) and precise localization (pixel)
    composite_score = (image_f1 * 0.5) + (pixel_f1 * 0.5)
    
    trial.report(composite_score, step=0)

    return composite_score


def run_optimization():
    """
    Initializes the Optuna study with a SQLite backend for the dashboard.
    Executes exactly 30 trials.
    """
    config_path = "config.yaml"
    base_config = load_config(config_path)

    # Define persistent storage for optuna-dashboard
    db_path = "sqlite:///efficientad_optuna.db"
    
    print(f"Starting EfficientAD optimization. Data will be saved to {db_path}")

    study = optuna.create_study(
        study_name="efficientad-optimization",
        storage=db_path,
        load_if_exists=True, # Resumes the study if interrupted
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Fixed to 30 trials as requested
    study.optimize(lambda trial: objective(trial, base_config), n_trials=30)

    print("\n=== Optimization Finished ===")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Score: {study.best_value:.4f}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    run_optimization()    