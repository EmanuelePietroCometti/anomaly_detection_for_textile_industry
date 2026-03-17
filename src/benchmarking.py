import time
import torch
import optuna
import pandas as pd
from functools import partial
from torchvision.transforms.v2 import Compose, Resize, Normalize, ToTensor

from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.pre_processing import PreProcessor
from config import load_config

from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.lightning_model import EfficientAdModelSize

def objective_patchcore(trial, dataset_cfg, gen_config, custom_transform):
    """
    Optuna objective function for PatchCore.
    Evaluates a set of hyperparameters by training the model and benchmarking inference time.
    """
    backbone_choice = trial.suggest_categorical("backbone", ["efficientnet_b4", "resnet18", "wide_resnet50_2"])

    resnet_layers_options = {
        "low_level": ["layer1", "layer2"],
        "mid_level": ["layer2", "layer3"], 
        "high_level": ["layer3", "layer4"],
        "multi_level": ["layer1", "layer2", "layer3"],
    }

    imagenet_layers_options = {
        "low_level": ["blocks.0", "blocks.1"],
        "mid_level": ["blocks.2", "blocks.4"],
        "mid_high_level": ["blocks.1", "blocks.2"],
        "high_level": ["blocks.4", "blocks.6"], 
        "multi_level": ["blocks.2", "blocks.3", "blocks.4"]
    }
    
    # Selection logic remains identical
    if backbone_choice == "efficientnet_b4":
        layer_key = trial.suggest_categorical("eff_b4_layer_type", ["low_level", "mid_level", "mid_high_level", "high_level", "multi_level"])
        layers_choice = imagenet_layers_options[layer_key]
    else:
        layer_key = trial.suggest_categorical("resnet_layer_type", ["low_level", "mid_level", "high_level", "multi_level"])
        layers_choice = resnet_layers_options[layer_key]
    
    n_neighbors = trial.suggest_int("num_nearest_neighbors", 3, 15)
    coreset_ratio = trial.suggest_float("coreset_sampling_ratio", 0.01, 0.2, log=True)
    
    # MODEL & ENGINE
    pre_processor = PreProcessor(transform=custom_transform)
    model = Patchcore(
        backbone=backbone_choice, 
        layers=layers_choice, 
        num_neighbors=n_neighbors, 
        coreset_sampling_ratio=coreset_ratio,
        pre_processor=pre_processor
    )
    
    datamodule = Folder(
        name="textiles_ead",
        root=dataset_cfg.get("root", "./data"),
        normal_dir=dataset_cfg.get("train_path", "./data/train").replace("./data/", ""),
        abnormal_dir=dataset_cfg.get("test_reject_path", "./data/test/reject").replace("./data/", ""),
        normal_test_dir=dataset_cfg.get("test_good_path", "./data/test/good").replace("./data/", ""),
        extensions=tuple(gen_config.get("valid_extensions", [".bmp", ".BMP"])),
        train_batch_size=1,
        eval_batch_size=1,
    )
    
    engine = Engine(accelerator="gpu", devices=1)
    
    try:
        engine.fit(datamodule=datamodule, model=model)
        results = engine.test(datamodule=datamodule, model=model)[0]
        
        # EXTRACT METRICS
        image_auroc = results.get("image_AUROC", 0.0)
        
        # Save extra metrics to trial attributes for the final table
        trial.set_user_attr("F1", f"{results.get('image_F1Score', 0.0):.4f}")
        
        # INFERENCE BENCHMARK
        model.eval()
        device = next(model.parameters()).device
        datamodule.setup(stage="test")
        real_batch = next(iter(datamodule.test_dataloader()))
        real_image = real_batch["image"][0].unsqueeze(0).to(device)
        
        with torch.no_grad():
            for _ in range(10): _ = model(real_image)
            torch.cuda.synchronize()
            t_start = time.perf_counter() # time.perf_counter() return the time in seconds
            for _ in range(50): _ = model(real_image)
            torch.cuda.synchronize()
            avg_inference_ms = ((time.perf_counter() - t_start) / 50) * 1000
            
        return image_auroc, avg_inference_ms
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()

def run_benchmarking_patchcore(n_trials=50):
    """
    Initializes configurations, creates a multi-objective Optuna study, 
    and executes the optimization process for PatchCore.
    """
    # Load config and define transform
    config = load_config()
    paths = config.get("paths", {})
    
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    objective_with_args = partial(
        objective_patchcore, 
        dataset_cfg=paths, 
        gen_config=config.get("general_configuration", {}),
        custom_transform=transform
    )
    
    study = optuna.create_study(
        study_name="Patchcore_Optimization",
        directions=["maximize", "minimize"], 
        sampler=optuna.samplers.NSGAIISampler()
    )
    study.optimize(objective_with_args, n_trials)
    
    print("\n" + "="*100)
    print("FULL BENCHMARK RESULTS (ALL COMPLETED TRIALS)")
    print("="*100)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    results_data = []
    
    for trial in completed_trials:
        row = {
            "Trial_ID": trial.number,
            "AUROC": f"{trial.values[0]:.4f}",
            "Latency (ms)": f"{trial.values[1]:.2f}",
            "F1": trial.user_attrs.get("F1", "0.0000"),
            **trial.params
        }
        results_data.append(row)

    if not results_data:
        print("\n[WARNING] No trials were completed successfully. Check the error logs.")
    else:
        df_results = pd.DataFrame(results_data)

        metric_cols = ["Trial_ID", "AUROC", "F1", "Latency (ms)"]
        param_cols = [c for c in df_results.columns if c not in metric_cols]
        
        df_final = df_results[metric_cols + param_cols].sort_values(by="AUROC", ascending=False)

        print(df_final.to_string(index=False))

        output_path = config["paths"].get("benchmark_pareto_front_patchcore", "results/benchmark_pareto_front_patchcore.csv")
        df_final.to_csv(output_path, index=False)
        print(f"\nFull benchmark history exported to: {output_path}")

def objective_efficientad(trial, dataset_cfg, gen_config, custom_transform):
    """
    Optuna objective function for EfficientAD.
    Evaluates a set of hyperparameters by training the model and benchmarking inference time.
    """
    model_size_choice = trial.suggest_categorical("model_size", ["S", "M"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    pre_processor = PreProcessor(transform=custom_transform)
    enum_size = EfficientAdModelSize.S if model_size_choice == "S" else EfficientAdModelSize.M
    
    model = EfficientAd(
        model_size=enum_size,
        lr=learning_rate,
        weight_decay=weight_decay,
        pre_processor=pre_processor
    )
    
    datamodule = Folder(
        name="textiles_ead",
        root=dataset_cfg.get("root", "./data"),
        normal_dir=dataset_cfg.get("train_path", "./data/train").replace("./data/", ""),
        abnormal_dir=dataset_cfg.get("test_reject_path", "./data/test/reject").replace("./data/", ""),
        normal_test_dir=dataset_cfg.get("test_good_path", "./data/test/good").replace("./data/", ""),
        extensions=tuple(gen_config.get("valid_extensions", [".bmp", ".BMP"])),
        train_batch_size=1,
        eval_batch_size=8
    )
    max_epochs = gen_config.get("max_epochs", 50)
    engine = Engine(
        accelerator="cuda" if torch.cuda.is_available() else "cpu", 
        devices=1,
        max_epochs=max_epochs,
        precision=16 if torch.cuda.is_available() else 32,
    )
    
    try:
        # Fit and Test
        engine.fit(datamodule=datamodule, model=model)
        results = engine.test(datamodule=datamodule, model=model)[0]
        
        # Extract Performance Metrics
        image_auroc = results.get("image_AUROC", 0.0)
        trial.set_user_attr("F1", f"{results.get('image_F1Score', 0.0):.4f}")
        
        # INFERENCE BENCHMARK
        model.eval()
        device = next(model.parameters()).device
        datamodule.setup(stage="test")
        real_batch = next(iter(datamodule.test_dataloader()))
        real_image = real_batch["image"][0].unsqueeze(0).to(device)
        
        with torch.no_grad():
            for _ in range(10): _ = model(real_image)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            
            # Measurement Phase
            t_start = time.perf_counter()
            for _ in range(50): _ = model(real_image)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            
            avg_inference_ms = ((time.perf_counter() - t_start) / 50) * 1000
            
        return image_auroc, avg_inference_ms
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


def run_benchmarking_efficientad(n_trials=30):
    """
    Initializes configurations, creates a multi-objective Optuna study, 
    and executes the optimization process. Saves all trials to CSV.
    """

    config = load_config()
    paths = config.get("paths", {})
   
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
    ])
    
    # CREATE MULTI-OBJECTIVE STUDY
    study = optuna.create_study(
        study_name="EfficientAD_Optimization",
        directions=["maximize", "minimize"], 
        sampler=optuna.samplers.NSGAIISampler() # Standard sampler for multi-objective
    )
    
    # Use functools.partial to pass extra arguments to the objective
    objective_with_args = partial(
        objective_efficientad,
        dataset_cfg=paths, 
        gen_config=config.get("general_configuration", {}), 
        custom_transform=transform
    )
    
    print(f"Starting Optuna study for EfficientAD with {n_trials} trials...")
    study.optimize(objective_with_args, n_trials=n_trials)
    
    print("\n" + "="*100)
    print("FULL BENCHMARK RESULTS (ALL COMPLETED TRIALS)")
    print("="*100)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    results_data = []
    
    for trial in completed_trials:
        row = {
            "Trial_ID": trial.number,
            "AUROC": f"{trial.values[0]:.4f}",
            "Latency (ms)": f"{trial.values[1]:.2f}",
            "F1": trial.user_attrs.get("F1", "0.0000"),
            **trial.params
        }
        results_data.append(row)

    if not results_data:
        print("\n[WARNING] No trials were completed successfully. Check the error logs.")
    else:
        df_results = pd.DataFrame(results_data)

        metric_cols = ["Trial_ID", "AUROC", "F1", "Latency (ms)"]
        param_cols = [c for c in df_results.columns if c not in metric_cols]
        
        df_final = df_results[metric_cols + param_cols].sort_values(by="AUROC", ascending=False)

        print(df_final.to_string(index=False))

        output_path = config["paths"].get("benchmark_pareto_front_efficientad", "results/benchmark_pareto_front_efficientad.csv")
        df_final.to_csv(output_path, index=False)
        print(f"\nFull benchmark history exported to: {output_path}")

if __name__ == "__main__":  
    run_benchmarking_patchcore(n_trials=30)
    run_benchmarking_efficientad(n_trials=10)