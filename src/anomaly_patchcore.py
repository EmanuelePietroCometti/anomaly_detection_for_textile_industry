import os
import logging
import warnings
import torch
import wandb
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from anomalib import TaskType
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from anomalib.loggers import AnomalibWandbLogger
from anomalib.deploy import ExportType
from torchvision.transforms.v2 import Compose, Resize, ToTensor, Normalize
from config import load_config

def apply_patchcore_model(backbone="efficientnet_b5", layers=["blocks.4", "blocks.6"], custom_weights_path=None):
    """
    Applies the Patchcore model. Loads custom weights if provided.
    Patchcore is a memory-bank model, so no multi-epoch training is required.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightning.*")
    warnings.filterwarnings("ignore", message=".*persistent_workers=True.*")
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    
    torch.set_float32_matmul_precision('medium')
    
    # ENVIRONMENT SETUP
    load_dotenv()
    if os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    if os.getenv("HUGGING_FACE_HUB_TOKEN"):
        login(token=os.getenv("HUGGING_FACE_HUB_TOKEN"))
        
    print("Loading configurations...")
    config = load_config("config.yaml")
    patchcore_cfg = config.get("patchcore_configuration", {})
    datamodule_cfg = config.get("datamodule_configruation", {})
    gen_config = config.get("general_configuration", {})
    
    custom_transform = Compose([
        Resize((224, 224)), 
        ToTensor(), 
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
        eval_batch_size=patchcore_cfg.get("eval_batch_size", 32) 
    )
    datamodule.setup()

    # MODEL INITIALIZATION 
    print("Initializing Patchcore model...")
    model = Patchcore(
        backbone=backbone,
        layers=layers,
        coreset_sampling_ratio=patchcore_cfg.get("coreset_sampling_ratio", 0.1),
        num_neighbors=patchcore_cfg.get("num_nearest_neighbors", 9),
    )

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
        callbacks=[checkpoint_callback, TimerCallback()],
        transform = custom_transform
    )

    # FEATURE EXTRACTION
    print("Extracting features for Patchcore memory bank...")
    engine.fit(model=model, datamodule=datamodule)
    
    # EVALUATION
    print("Evaluating model...")
    engine.test(model=model, datamodule=datamodule)
    engine.trainer.save_checkpoint("checkpoints/patchcore-tested.ckpt")
    
    # METRICS EXTRACTION
    print("\nExtracting predictions for the Confusion Matrix...")
    predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader())

    y_true, y_pred = [], []
    for batch in predictions:
        y_true.extend(batch.gt_label.cpu().numpy())
        y_pred.extend(batch.pred_label.cpu().numpy())

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n{'='*65}\n DETAILED REPORT: CONFUSION MATRIX\n{'='*65}")
    print(f"                            | True: GOOD     | True: DEFECT ")
    print(f"-" * 65)
    print(f" Predicted: GOOD          | [ TN: {tn:<4} ]    | [ FN: {fn:<4} ] ")
    print(f" Predicted: DEFECT        | [ FP: {fp:<4} ]    | [ TP: {tp:<4} ] ")
    print(f"{'='*65}")
    print(f"    Accuracy    : {acc*100:>6.2f}%")
    print(f"    Precision   : {prec*100:>6.2f}% (Valid defects when rejected)")
    print(f"    Recall      : {rec*100:>6.2f}% (True defects found)")
    print(f"    F1-Score    : {f1:>6.4f}\n{'='*65}\n")

    if fn > 0:
        print(f"WARNING: Model generated {fn} False Negatives (Missed defects).")
    if fp > 0:
        print(f"NOTE: Model generated {fp} False Positives (Good parts rejected).")

def export_checkpoint_to_onnx(ckpt_path, export_dir):
    """
    Exports the trained PatchCore model to ONNX.
    """
    config = load_config()
    model_arch = config["model_architecture"]
    img_size = tuple(config["general_configuration"]["image_size"])

    model = Patchcore(backbone=model_arch["backbone"], layers=model_arch["layers"])
    engine = Engine()

    print(f"Starting ONNX export with input size {img_size}...")
    try:
        # Simplified parameters to ensure stability with the internal ONNX wrapper
        export_path = engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=export_dir,
            ckpt_path=ckpt_path,
            input_size=img_size
        )
        print(f"\n[SUCCESS] Model exported to: {export_path}")
    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}")

if __name__ == "__main__":
    # Removed the duplicated block to prevent double execution calls
    config = load_config()
    export_checkpoint_to_onnx(
        ckpt_path=config["paths"]["checkpoint_destination"], 
        export_dir=config["paths"]["exports_onnx_path"]
    )