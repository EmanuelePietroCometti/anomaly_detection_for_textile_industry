import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, auc
)
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.loggers import AnomalibWandbLogger
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from utils import backup_and_cleanup_latest_run 

def run_anomaly_pipeline(model, config, project_name="anomaly-pipeline"):
    """
    Unified pipeline to handle data loading, training, and evaluation 
    for ANY initialized Anomalib model.
    """
    datamodule_cfg = config.get("datamodule_configruation", {})
    gen_config = config.get("general_configuration", {})
    paths = config.get("paths", {})
    model_arch = config.get("model_architecture", {})
    
    # Extract dynamic model details
    model_name = model.__class__.__name__
    backbone = model_arch.get("backbone", "custom_model")
    layers = model_arch.get("layers", ["custom_layers"])
    layers_str = "_".join(layers) if isinstance(layers, list) else str(layers)
    timestamp = config.get("global_timestamp", "latest")
    
    # Setup Datamodule
    datamodule = Folder(
        name="textiles_dataset",
        root=datamodule_cfg.get("root", "./data"),
        normal_dir=datamodule_cfg.get("train_dir", "train/good"),
        abnormal_dir=datamodule_cfg.get("test_dir_reject", "test/reject"),
        normal_test_dir=datamodule_cfg.get("test_dir_good", "test/good"),
        extensions=tuple(gen_config.get("valid_extensions", [".bmp"])),
        train_batch_size=datamodule_cfg.get("train_batch_size", 32),
        eval_batch_size=datamodule_cfg.get("eval_batch_size", 32),
    )
    datamodule.setup()

    # Setup Engine
    logger = AnomalibWandbLogger(project=project_name, name=model_name)
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename=f"{model_name}-latest")
    
    # Dynamically fetch epochs (e.g., 1 for Patchcore, 100 for EfficientAD)
    max_epochs = config.get(f"{model_name.lower()}_settings", {}).get("num_epochs", 1)
    
    engine = Engine(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, TimerCallback()],
        accelerator="auto"
    )

    # Fit and Test
    print(f"\n--- Training {model_name} ---")
    engine.fit(model=model, datamodule=datamodule)
    
    print(f"\n--- Evaluating {model_name} ---")
    engine.test(model=model, datamodule=datamodule)

    # Universal Metrics Extraction
    print("\nExtracting predictions for Metrics and AUROC...")
    predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader())
    
    y_true, y_pred, y_scores = [], [], []
    for batch in predictions:
        y_true.extend(batch.gt_label.cpu().numpy())
        y_pred.extend(batch.pred_label.cpu().numpy())
        y_scores.extend(batch.pred_score.cpu().numpy())

    # Confusion Matrix & Report
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    report_text = (
        f"\n{'='*65}\n DETAILED REPORT: CONFUSION MATRIX\n{'='*65}\n"
        f"                            | Predicted: GOOD     | Predicted: DEFECT \n"
        f"{'-'*65}\n"
        f" True: GOOD          | [ TN: {tn:<4} ]    | [ FP: {fp:<4} ] \n"
        f" True: DEFECT        | [ FN: {fn:<4} ]    | [ TP: {tp:<4} ] \n"
        f"{'='*65}\n"
        f"    Accuracy    : {acc*100:>6.2f}%\n"
        f"    Precision   : {prec*100:>6.2f}% (Valid defects when rejected)\n"
        f"    Recall      : {rec*100:>6.2f}% (True defects found)\n"
        f"    F1-Score    : {f1:>6.4f}\n"
        f"{'='*65}\n"
    )
    print(report_text)

    # Save Report
    report_filename = f"{timestamp}_evaluation_{model_name}_{backbone}.txt"
    report_dir = paths.get("report_path", f"results/{model_name}/report")
    os.makedirs(report_dir, exist_ok=True)
    report_filepath = os.path.join(report_dir, report_filename)

    with open(report_filepath, "w", encoding="utf-8") as file:
        file.write(report_text)
    print(f"[SUCCESS] Report successfully saved to: {report_filepath}")

    # AUROC Plot Generation
    print("Generating AUROC Curve...")
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} ROC (AUC = {roc_auc:.4f})')
    plt.fill_between(fpr, tpr, alpha=0.15, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    
    # Generic title that works for any model
    plt.title(
        f'Receiver Operating Characteristic - {model_name}\n'
        f'Architecture Details: {backbone} | {layers_str}\n',
        fontsize=10
    )
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)

    results_dir = Path(paths.get("auroc_path", f"results/{model_name}/auroc"))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{timestamp}_{model_name}_auroc_{backbone}_{layers_str}.png"
    plot_path = results_dir / filename
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] AUROC Curve exported successfully to: {plot_path}")
    
    # Backup and Cleanup (ensure this is imported properly)
    symlink_path = paths.get("symlink_path")
    anomaly_images = paths.get("anomaly_images")
    if symlink_path and anomaly_images:
        try:
            # Assuming backup_and_cleanup_latest_run is available in your scope
            backup_and_cleanup_latest_run(symlink_path, anomaly_images, backbone, layers, config)
        except NameError:
            print("[WARNING] 'backup_and_cleanup_latest_run' is not defined. Skipping cleanup.")
    
    return engine