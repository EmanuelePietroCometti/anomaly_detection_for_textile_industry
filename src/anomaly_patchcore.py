import os
import logging
import warnings
import torch
import wandb
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from anomalib.data import Folder
from anomalib.data.utils.split import ValSplitMode, TestSplitMode
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from anomalib.loggers import AnomalibWandbLogger
from anomalib.deploy import ExportType
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize, RandomCrop, CenterCrop
from config import load_config
from lightning.pytorch import seed_everything
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import shutil
from pathlib import Path

logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric").setLevel(logging.ERROR)
seed_everything(42, workers=True)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightning.*")
warnings.filterwarnings("ignore", message=".*persistent_workers=True.*")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
torch.set_float32_matmul_precision('medium')

import torch
from anomalib.metrics import F1AdaptiveThreshold

import torch
from anomalib.metrics import F1AdaptiveThreshold

class TargetRecallThreshold(F1AdaptiveThreshold):
    """
    Custom Adaptive Threshold that guarantees a minimum target Recall 
    while maximizing Precision to minimize false positives (scraps).
    """
    def __init__(self, target_recall=0.99, default_value=0.5, **kwargs):
        fields = kwargs.pop("fields", ["pred_score", "gt_label"])
        
        super().__init__(fields=fields, **kwargs)
        self.target_recall = target_recall

    def compute(self) -> torch.Tensor:
        # Compute Precision, Recall, and all possible Thresholds from the PR curve
        precision, recall, thresholds = self.precision_recall_curve.compute()
        
        # Find indices where the model achieves the requested target Recall
        valid_indices = torch.where(recall >= self.target_recall)[0]
        
        if len(valid_indices) > 0:
            # Filter precision and thresholds using only the safe indices
            valid_precisions = precision[valid_indices]
            valid_thresholds = thresholds[valid_indices]
            
            # Pick the index with the highest Precision to minimize scraps
            best_idx = torch.argmax(valid_precisions)
            self.value = valid_thresholds[best_idx]
        else:
            # Fallback: If the target is mathematically unreachable, 
            # pick the threshold that yields the absolute maximum Recall possible.
            print(f"\n[WARNING] Target recall {self.target_recall} unreachable. Defaulting to max possible recall.")
            best_idx = torch.argmax(recall)
            self.value = thresholds[best_idx]
            
        return self.value

def backup_and_cleanup_latest_run(symlink_path, dest_parent_dir, backbone, layers, config):
    """
    Resolves a symlink copies its target directory to a new 
    custom-named folder, and safely deletes both the original target and the symlink.
    """
    dataset_cfg = config["dataset_pipeline"]
    dataset_version = dataset_cfg["dataset_version"]
    if not os.path.islink(symlink_path):
        print(f"[ERROR] '{symlink_path}' is not a valid symbolic link.")
        return

    real_source_dir = Path(symlink_path)
    print(f"Symlink resolved. Actual source directory: {real_source_dir}")

    timestamp = config["global_timestamp"]
    layers_str = "_".join(layers) if isinstance(layers, list) else str(layers)
    new_folder_name = f"{timestamp}_{backbone}_{layers_str}_d{dataset_version}"
    destination_dir = Path(dest_parent_dir) / new_folder_name

    try:
        print(f"Copying files to: {destination_dir}")
        shutil.copytree(real_source_dir, destination_dir, dirs_exist_ok=True)
        print("[SUCCESS] Backup completed successfully.")
        print(f"Deleting original directory: {real_source_dir}")
        shutil.rmtree(real_source_dir)
        print(f"Removing symlink: {symlink_path}")
        os.unlink(symlink_path)
        print("[SUCCESS] Cleanup completed. Only the customized backup remains.")

    except Exception as e:
        print(f"[ERROR] Process failed: {e}. Original files were NOT deleted.")

def apply_patchcore_model(config, backbone="efficientnet_b5", layers=["blocks.4", "blocks.6"], custom_weights_path=None):
    """
    Applies the Patchcore model. Loads custom weights if provided.
    Patchcore is a memory-bank model, so no multi-epoch training is required.
    """
    # ENVIRONMENT SETUP
    load_dotenv()
    if os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    if os.getenv("HUGGING_FACE_HUB_TOKEN"):
        login(token=os.getenv("HUGGING_FACE_HUB_TOKEN"))
        
    print("Loading configurations...")
    patchcore_cfg = config.get("patchcore_configuration", {})
    datamodule_cfg = config.get("datamodule_configruation", {})
    gen_config = config.get("general_configuration", {})
    paths = config.get("paths", {})
    
    image_size = gen_config["image_size"]
    crop_size = gen_config["crop_size"]

    if "efficientnet" in backbone:
        transform = Compose([
            Resize(crop_size, interpolation=InterpolationMode.BICUBIC), 
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = Compose([
            Resize(image_size),
            CenterCrop(crop_size),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    datamodule = Folder(
        name="textiles",
        root=datamodule_cfg.get("root", "./data"),
        normal_dir=datamodule_cfg.get("train_dir", "./data/dataset_patchcore/train").replace("./data/", ""),
        abnormal_dir=datamodule_cfg.get("test_dir_reject", "./data/dataset_patchcore/test/reject").replace("./data/", ""),
        normal_test_dir=datamodule_cfg.get("test_dir_good", "./data/dataset_patchcore/test/good").replace("./data/", ""),
        extensions=tuple(gen_config.get("valid_extensions", [".bmp",".BMP"])),
        train_batch_size=patchcore_cfg.get("train_batch_size", 32),
        eval_batch_size=patchcore_cfg.get("eval_batch_size", 32),
        val_split_mode=ValSplitMode.FROM_TEST,
        val_split_ratio=0.3,
        test_split_mode=TestSplitMode.FROM_DIR,
        seed=datamodule_cfg.get("seed", 42)
    )
    datamodule.setup()

    # MODEL INITIALIZATION 
    print("Initializing Patchcore model...")
    coreset_sampling_ratio=patchcore_cfg.get("coreset_sampling_ratio", 0.1)
    num_neighbors=patchcore_cfg.get("num_nearest_neighbors", 9)
    model = Patchcore(
        backbone=backbone,
        layers=layers,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
    )

    model.transform = transform
    model.image_threshold = TargetRecallThreshold(target_recall=0.99)
    model.pixel_threshold = TargetRecallThreshold(target_recall=0.99)

    # CUSTOM WEIGHTS INJECTION
    if custom_weights_path and os.path.exists(custom_weights_path):
        print(f"Loading custom weights from: {custom_weights_path}")
        state_dict = torch.load(custom_weights_path, map_location="cpu")
        
        anomalib_state_dict = model.state_dict()
        adapted_state_dict = {}
        
        for custom_key, tensor in state_dict.items():
            for anomalib_key in anomalib_state_dict.keys():
                if custom_key in anomalib_key:
                    adapted_state_dict[anomalib_key] = tensor
                    break
                    
        model.load_state_dict(adapted_state_dict, strict=False)
        print(f"Custom weights injected. Adapted layers: {len(adapted_state_dict)}/{len(state_dict)}")
    else:
        print("Using default ImageNet weights for backbone.")

    
    logger = AnomalibWandbLogger(project="anomaly-REDA", name="patchcore-run")
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename="patchcore-latest")
    
    engine = Engine(
        logger=logger,
        accelerator="gpu",
        callbacks=[checkpoint_callback, TimerCallback()]
    )

    # FEATURE EXTRACTION
    print("Extracting features for Patchcore memory bank...")
    engine.fit(model=model, datamodule=datamodule)
    
    # EVALUATION
    print("Evaluating model...")
    engine.test(model=model, datamodule=datamodule)
    engine.trainer.save_checkpoint("checkpoints/patchcore-tested.ckpt")
    
    # METRICS EXTRACTION
    print("\nExtracting predictions for Metrics and AUROC...")
    predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader())

    y_true, y_pred, y_scores = [], [], []
    for batch in predictions:
        y_true.extend(batch.gt_label.cpu().numpy())
        # Binary prediction (0 = Good, 1 = Defect)
        y_pred.extend(batch.pred_label.cpu().numpy())
        y_scores.extend(batch.pred_score.cpu().numpy()) 

    # CONFUSION MATRIX
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
    if fn > 0:
        print(f"WARNING: Model generated {fn} False Negatives (Missed defects).")
    if fp > 0:
        print(f"NOTE: Model generated {fp} False Positives (Good parts rejected).")
    timestamp = config["global_timestamp"]
    report_filename = f"{timestamp}_evaluation_report_{backbone}_{layers}.txt"
    report_dir = paths.get("report_path", "results/Patchcore/report")
    os.makedirs(report_dir, exist_ok=True)
    report_filepath = os.path.join(report_dir, report_filename)

    with open(report_filepath, "w", encoding="utf-8") as file:
        file.write(report_text)

    print(f"[SUCCESS] Report successfully saved to: {report_filepath}")
    

    # AUROC PLOT GENERATION
    print("Generating AUROC Curve...")
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    layers_str = "_".join(layers) if isinstance(layers, list) else str(layers)
    layers_title = ", ".join(layers) if isinstance(layers, list) else str(layers)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Patchcore ROC (AUC = {roc_auc:.4f})')
    plt.fill_between(fpr, tpr, alpha=0.15, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    
    plt.title(
        f'Receiver Operating Characteristic\n'
        f'Backbone: {backbone} | Layers: {layers_title}\n'
        f'Coreset Ratio: {coreset_sampling_ratio} | Neighbors: {num_neighbors}\n',
        fontsize=10
    )
    
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)

    results_dir = Path(config["paths"]["auroc_path"])
    os.makedirs(results_dir, exist_ok=True)
    timestamp = config["global_timestamp"]
    filename = f"{timestamp}_patchcore_auroc_{backbone}_cr{coreset_sampling_ratio}_nn{num_neighbors}_{layers_str}.png"
    plot_path = results_dir / filename
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] AUROC Curve exported successfully to: {plot_path}")
    anomaly_images = paths.get("anomaly_images")
    symlink_path = paths["symlink_path"]
    backup_and_cleanup_latest_run(symlink_path, anomaly_images, backbone, layers, config)

def export_checkpoint_to_onnx(ckpt_path, export_dir, config):
    """
    Exports the trained PatchCore model to ONNX.
    """
    model_arch = config["model_architecture"]
    img_size = tuple(config["general_configuration"]["crop_size"]) ## I used the crop size because the effective size of the input images is the crop size
    backbone=model_arch["backbone"]
    layers=model_arch["layers"]

    model = Patchcore(backbone=backbone, layers=layers)
    engine = Engine()

    print(f"Starting ONNX export with input size {img_size}...")
    try:
        # Simplified parameters to ensure stability with the internal ONNX wrapper
        export_path = engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=export_dir,
            ckpt_path=ckpt_path,
            input_size=img_size,
            onnx_kwargs={
                "dynamo": False,
            }
        )
        timestamp = config["global_timestamp"]
        directory, original_filename = os.path.split(export_path)
        name, extension = os.path.splitext(original_filename)
    
        # Convert list to a safe string format
        layers_str = "_".join(layers) if isinstance(layers, list) else str(layers)
        
        # Build the new filename safely
        new_filename = f"{timestamp}_{name}_{backbone}_{layers_str}{extension}"
        new_export_path = os.path.join(directory, new_filename)
        os.rename(export_path, new_export_path)
        
        print(f"\n[SUCCESS] Model exported to: {export_path}")
    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}")

if __name__ == "__main__":
    config = load_config()
    export_checkpoint_to_onnx(
        ckpt_path=config["paths"]["checkpoint_destination"], 
        export_dir=config["paths"]["exports_onnx_path"]
    )