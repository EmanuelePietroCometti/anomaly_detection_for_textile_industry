"""
Hyperparameter Optimization (HPO) for PatchCore using Optuna.
Saves trials to a local SQLite database to enable real-time tracking 
via optuna-dashboard.
"""

import os
import gc
import time
import torch
import optuna
import tempfile
from pathlib import Path

from anomalib.data import Folder
from anomalib.data.utils.split import ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore

from config import load_config

def objective(trial, base_config):
    """
    Objective function for Optuna. Evaluates PatchCore performance
    balancing defect detection and dust tolerance.
    """
    # Define the Search Space
    backbone = trial.suggest_categorical("backbone", ["resnet18", "resnet50", "wide_resnet50_2"])
    
    layers_choice = trial.suggest_categorical("layers", ["l1_l2", "l2_l3", "l3_l4", "l1_l2_l3"])
    layer_map = {
        "l1_l2": ["layer1", "layer2"],
        "l2_l3": ["layer2", "layer3"],
        "l3_l4": ["layer3", "layer4"],
        "l1_l2_l3": ["layer1", "layer2", "layer3"]
    }
    selected_layers = layer_map[layers_choice]
    
    coreset_ratio = trial.suggest_float("coreset_sampling_ratio", 0.05, 0.25, step=0.05)
    n_neighbors = trial.suggest_int("num_nearest_neighbors", 3, 15, step=2)

    # Setup DataModule
    datamodule_cfg = base_config.get("datamodule_configuration", {})
    root_path = Path(datamodule_cfg.get("root", "./data/mvtec"))
    category = datamodule_cfg.get("category", "reda")
    dataset_path = root_path / category
    
    test_path = dataset_path / "test"
    abnormal_dirs = [f"test/{d}" for d in os.listdir(test_path) if (test_path / d).is_dir() and d != "good"] if test_path.exists() else []

    datamodule = Folder(
        name=category,
        root=str(dataset_path),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir=abnormal_dirs,
        mask_dir="ground_truth",
        train_batch_size=base_config["patchcore_configuration"].get("train_batch_size", 32),
        eval_batch_size=base_config["patchcore_configuration"].get("eval_batch_size", 32),
        num_workers=datamodule_cfg.get("num_workers", 4),
        val_split_mode=ValSplitMode.SAME_AS_TEST
    )

    # Model Initialization
    model = Patchcore(
        backbone=backbone,
        layers=selected_layers,
        coreset_sampling_ratio=coreset_ratio,
        num_neighbors=n_neighbors,
    )

    # Create a temporary directory that will automatically be cleaned up
    with tempfile.TemporaryDirectory() as tmp_dir:
        
        # Engine Setup
        engine = Engine(
            max_epochs=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=base_config["general_configuration"].get("precision", "32-true"),
            default_root_dir=tmp_dir,
            logger=False
        )

        # Training
        engine.fit(model=model, datamodule=datamodule)

        # Evaluation
        test_results = engine.test(model=model, datamodule=datamodule)

    # Explicit memory cleanup to prevent GPU VRAM saturation
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
    
    # Composite Score: 60% global (dust handling), 40% spatial (micro-defects)
    composite_score = (image_f1 * 0.6) + (pixel_f1 * 0.4)
    
    trial.report(composite_score, step=0)

    return composite_score

def run_optimization():
    """
    Initializes the Optuna study with a SQLite backend for the dashboard.
    """
    config_path = "config.yaml"
    base_config = load_config(config_path)

    # Define persistent storage for optuna-dashboard
    db_path = "sqlite:///patchcore_optuna.db"
    
    print(f"Starting optimization. Data will be saved to {db_path}")

    study = optuna.create_study(
        study_name="patchcore-dust-optimization",
        storage=db_path,
        load_if_exists=True, # Resumes the study if interrupted
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(lambda trial: objective(trial, base_config), n_trials=30)

    print("\n=== Optimization Finished ===")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Score: {study.best_value:.4f}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    run_optimization()