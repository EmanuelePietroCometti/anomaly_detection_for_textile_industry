# src/anomaly_pipeline.py
import os
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.loggers import AnomalibWandbLogger
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from src.visualization import save_evaluation_report, plot_auroc_curve

def run_anomaly_pipeline(model, config, project_name="anomaly-pipeline"):
    """
    Unified pipeline to handle data loading, training, and evaluation.
    """
    datamodule_cfg = config.get("datamodule_configuration", {})
    gen_config = config.get("general_configuration", {})
    model_arch = config.get("model_architecture", {})
    
    model_name = model.__class__.__name__
    backbone = model_arch.get("backbone", "custom_model")
    layers = model_arch.get("layers", ["custom_layers"])
    paths = config["paths"]
    layers_str = "_".join(layers) if isinstance(layers, list) else str(layers)
    
    datamodule = Folder(
        name="textiles_dataset",
        root=datamodule_cfg.get("root", "./data"),
        normal_dir=datamodule_cfg.get("train_dir", "train/good").replace("./data/", ""),
        abnormal_dir=datamodule_cfg.get("test_dir_reject", "test/reject").replace("./data/", ""),
        normal_test_dir=datamodule_cfg.get("test_dir_good", "test/good").replace("./data/", ""),
        extensions=tuple(gen_config.get("valid_extensions", [".bmp", ".BMP"])),
        train_batch_size=datamodule_cfg.get("train_batch_size", 32),
        eval_batch_size=datamodule_cfg.get("eval_batch_size", 32),
    )
    datamodule.setup()

    logger = AnomalibWandbLogger(project=project_name, name=model_name)
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename=f"{model_name}-latest")
    
    max_epochs = config.get(f"{model_name.lower()}_settings", {}).get("num_epochs", 1)
    
    engine = Engine(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, TimerCallback()],
        accelerator="auto"
    )

    print(f"\n--- Training {model_name} ---")
    engine.fit(model=model, datamodule=datamodule)
    
    print(f"\n--- Evaluating {model_name} ---")
    test_results = engine.test(model=model, datamodule=datamodule)

    print("\nExtracting predictions for Metrics and AUROC...")
    predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader())
    
    y_true, y_pred, y_scores = [], [], []
    
    for batch in predictions:
        if isinstance(batch, dict):
            gt = batch.get("gt_label", batch.get("label", []))
            pred = batch.get("pred_label", [])
            score = batch.get("pred_score", [])
        else:
            gt = getattr(batch, "gt_label", getattr(batch, "label", []))
            pred = getattr(batch, "pred_label", [])
            score = getattr(batch, "pred_score", [])
            
        if len(gt) > 0: y_true.extend(gt.cpu().numpy())
        if len(pred) > 0: y_pred.extend(pred.cpu().numpy())
        if len(score) > 0: y_scores.extend(score.cpu().numpy())

    if len(y_true) > 0:
        save_evaluation_report(y_true, y_pred, model_name, backbone, config)
        plot_auroc_curve(y_true, y_scores, model_name, backbone, layers_str, config)
    else:
        print("[WARNING] Ground truth labels were not found in predict step. Skipping plots.")
    
    return engine