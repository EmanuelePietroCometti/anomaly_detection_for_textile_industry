import time
import torch
import optuna
import pandas as pd
from functools import partial
from torchvision.transforms.v2 import Compose, Resize, Normalize

from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.pre_processing import PreProcessor
from config import load_config

def objective(trial, dataset_cfg, gen_config, custom_transform):
    backbone_choice = trial.suggest_categorical("backbone", ["resnet18", "wide_resnet50_2", "efficientnet_b4"])

    resnet_layers_options = {
        "low_level": ["layer1", "layer2"],
        "mid_level": ["layer2", "layer3"], 
        "high_level": ["layer3", "layer4"],
        "multi_level": ["layer1", "layer2", "layer3", "layer4"],
    }
    imagenet_layers_options = {
        "low_level": ["blocks.0", "blocks.1"],
        "mid_level": ["blocks.2", "blocks.4"],
        "mid_high_level": ["blocks.1", "blocks.2"],
        "high_level": ["blocks.4", "blocks.6"],
        "multi_level": ["blocks.2", "blocks.3", "blocks.4", "blocks.6"]
    }
    
    # Selection logic remains identical
    if backbone_choice == "efficientnet_b4":
        layer_key = trial.suggest_categorical("eff_b4_layer_type", ["low_level", "mid_level", "high_level", "multi_level"])
        layers_choice = imagenet_layers_options[layer_key]
    else:
        layer_key = trial.suggest_categorical("resnet_layer_type", ["low_level", "mid_level", "high_level", "multi_level"])
        layers_choice = resnet_layers_options[layer_key]
    
    n_neighbors = trial.suggest_int("num_nearest_neighbors", 3, 15)
    coreset_ratio = trial.suggest_float("coreset_sampling_ratio", 0.01, 0.3, log=True)
    
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
        trial.set_user_attr("Precision", f"{results.get('image_Precision', 0.0):.4f}")
        trial.set_user_attr("Recall", f"{results.get('image_Recall', 0.0):.4f}")
        trial.set_user_attr("Accuracy", f"{results.get('image_Accuracy', 0.0):.4f}")
        
        # INFERENCE BENCHMARK
        model.eval()
        device = next(model.parameters()).device
        datamodule.setup(stage="test")
        real_batch = next(iter(datamodule.test_dataloader()))
        real_image = real_batch["image"][0].unsqueeze(0).to(device)
        
        with torch.no_grad():
            for _ in range(10): _ = model(real_image)
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            for _ in range(50): _ = model(real_image)
            torch.cuda.synchronize()
            avg_inference_ms = ((time.perf_counter() - t_start) / 50) * 1000
            
        return image_auroc, avg_inference_ms
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()

if __name__ == "__main__":
    # Load config and define transform
    config = load_config()
    paths = config.get("paths", {})
    
    transform = Compose([
        Resize((256, 256)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    objective_with_args = partial(
        objective, 
        dataset_cfg=paths, 
        gen_config=config.get("generation", {}), 
        custom_transform=transform
    )
    
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective_with_args, n_trials=50)
    
    print("\n" + "="*50)
    print("PARETO FRONT BENCHMARK RESULTS")
    print("="*50)

    best_trials = study.best_trials
    results_data = []
    
    for i, trial in enumerate(best_trials):
        row = {
            "Option": i + 1,
            "AUROC": f"{trial.values[0]:.4f}",
            "Latency (ms)": f"{trial.values[1]:.2f}",
            "F1": trial.user_attrs.get("F1"),
            "Precision": trial.user_attrs.get("Precision"),
            "Accuracy": trial.user_attrs.get("Accuracy"),
            **trial.params
        }
        results_data.append(row)

    df_results = pd.DataFrame(results_data)

    metric_cols = ["Option", "AUROC", "F1", "Precision", "Recall", "Accuracy", "Latency (ms)"]
    param_cols = [c for c in df_results.columns if c not in metric_cols]
    
    df_final = df_results[metric_cols + param_cols].sort_values(by="AUROC", ascending=False)

    print("\n" + "="*100)
    print("PARETO FRONT BENCHMARK RESULTS")
    print("="*100)
    print(df_final.to_string(index=False))

    output_path = config["paths"].get("benchmark_pareto_front", "benchmark_results.csv")
    df_final.to_csv(output_path, index=False)
    print(f"\nResults exported to: {output_path}")