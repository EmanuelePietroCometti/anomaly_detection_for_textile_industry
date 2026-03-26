import os
from anomalib.data import Folder 
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from anomalib.loggers import AnomalibWandbLogger
from anomalib.data.utils import ValSplitMode
from dotenv import load_dotenv
import wandb
from huggingface_hub import login
import torch
import warnings
from config import load_config
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from anomalib.deploy import ExportType
import logging

def apply_patchcore_model(backbone="efficientnet_b5", layers=["blocks.4", "blocks.6"], custom_weights_path=None):
    """
    Applies the Patchcore model. If custom_weights_path is provided, it loads the backbone weights
    resulting from transfer learning before proceeding with feature extraction.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightning.*")
    warnings.filterwarnings("ignore", message=".*persistent_workers=True.*")
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    torch.set_float32_matmul_precision('medium')
    
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    if hf_token:
        login(token=hf_token)
        
    print("Applying Patchcore model...")
    config = load_config("config.yaml")
    patchcore_cfg = config.get("patchcore_configuration", {})
    datamodule_cfg = config.get("datamodule_configruation", {})
    gen_config = config.get("general_configuration", {})
    
    datamodule = Folder(
        name="textiles",
        root=datamodule_cfg.get("root", "./data"),
        normal_dir=datamodule_cfg.get("train_dir", "./data/dataset_patchcore/train").replace("./data/", ""),
        abnormal_dir=datamodule_cfg.get("test_dir_reject", "./data/dataset_patchcore/test/reject").replace("./data/", ""),
        normal_test_dir=datamodule_cfg.get("test_dir_good", "./data/dataset_patchcore/test/good").replace("./data/", ""),
        extensions=tuple(gen_config.get("valid_extensions", [".bmp",".BMP"])),
        image_size=tuple(gen_config.get("image_size", [256, 256])),
        train_batch_size=patchcore_cfg.get("train_batch_size", 32),
        eval_batch_size=patchcore_cfg.get("eval_batch_size", 32),
        val_split_mode=ValSplitMode.FROM_TEST,
        val_split_ratio=0.3
    )
    datamodule.setup()

    logger = AnomalibWandbLogger(project="anomaly-REDA", name="patchcore-run")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="patchcore-latest",
    )
    timer_callback = TimerCallback()
    
    print("Initializing Patchcore model...")
    model = Patchcore(
        backbone=backbone,
        layers=layers,
        coreset_sampling_ratio=patchcore_cfg.get("coreset_sampling_ratio", 0.1),
        num_neighbors=patchcore_cfg.get("num_nearest_neighbors", 9),
    )

    # --- WEIGHTS INTEGRATION ---
    if custom_weights_path and os.path.exists(custom_weights_path):
        print(f"Loading custom weights from: {custom_weights_path}")
        state_dict = torch.load(custom_weights_path, map_location="cpu")
        
        # Extract expected keys from Anomalib
        anomalib_state_dict = model.state_dict()
        adapted_state_dict = {}
        
        # Dynamic key mapping (Standard PyTorch -> Anomalib Wrapper)
        for custom_key, tensor in state_dict.items():
            for anomalib_key in anomalib_state_dict.keys():
                # If the original key (e.g. 'conv1.weight') is contained in the anomalib one
                # (e.g. 'model.feature_extractor.backbone.conv1.weight'), perform the assignment
                if custom_key in anomalib_key:
                    adapted_state_dict[anomalib_key] = tensor
                    break
                    
        # Loading with strict=False (ignores FC/ProjectionHead layers that PatchCore doesn't use)
        missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
        print(f"Custom weights injected. Adapted layers: {len(adapted_state_dict)}/{len(state_dict)}")
    else:
        print("Custom weights not found or path not provided. Using default ImageNet weights.")

    engine = Engine(
        max_epochs=patchcore_cfg["num_epochs"],          
        logger=logger,
        accelerator="gpu",
        callbacks=[checkpoint_callback, timer_callback]
    )

    print("Starting training with Patchcore model...")
    engine.fit(model=model, datamodule=datamodule)
    print("Training completed.")

    print("Evaluating Patchcore model...")
    engine.test(model=model, datamodule=datamodule)
    print("Evaluation completed.")
    
    engine.trainer.save_checkpoint("checkpoints/patchcore-tested.ckpt")
    
    print("\nExtracting predictions for the Confusion Matrix...")
    predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader())

    y_true = []
    y_pred = []

    for batch in predictions:
        y_true.extend(batch.gt_label.cpu().numpy())
        y_pred.extend(batch.pred_label.cpu().numpy())

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n" + "="*65)
    print(" DETAILED REPORT: CONFUSION MATRIX")
    print("="*65)
    print(f"                            | True: GOOD     | True: DEFECT ")
    print("-" * 65)
    print(f" Predicted: GOOD          | [ TN: {tn:<4} ]    | [ FN: {fn:<4} ] ")
    print(f" Predicted: DEFECT        | [ FP: {fp:<4} ]    | [ TP: {tp:<4} ] ")
    print("="*65)
    print(f"    Accuracy    : {acc*100:>6.2f}%")
    print(f"    Precision   : {prec*100:>6.2f}% (When it rejects, it is an actual defect {prec*100:.0f}% of the time)")
    print(f"    Recall      : {rec*100:>6.2f}% (Finds {rec*100:.0f}% of all real defects)")
    print(f"    F1-Score    : {f1:>6.4f}")
    print("="*65 + "\n")

    if fn > 0:
        print(f"WARNING: The model generated {fn} False Negatives (Missed defects shipped to customer!).")
    if fp > 0:
        print(f"NOTE: The model generated {fp} False Positives (Good parts rejected/scrapped).")

def export_checkpoint_to_onnx(ckpt_path, export_dir):
    """
    Exports the PatchCore model to ONNX format using the official Anomalib API.
    """
    print(f"Loading configuration and initializing model...")
    config = load_config()
    model_arch = config["model_architecture"]
    
    img_size = tuple(config["general_configuration"]["image_size"])

    model = Patchcore( 
        backbone=model_arch["backbone"], 
        layers=model_arch["layers"]
    )
    
    engine = Engine()

    onnx_params = {
        "input_names": ["input"],
        "output_names": ["score", "anomaly_map"],
        "opset_version": 14,
        "dynamo": False
    }

    print(f"Starting official ONNX export with input size {img_size}...")
    try:
        export_path = engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=export_dir,
            ckpt_path=ckpt_path,
            input_size=img_size,
            onnx_kwargs=onnx_params
        )
        print(f"\n[SUCCESS] Model exported to: {export_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}")

if __name__ == "__main__":
    config = load_config()
    export_checkpoint_to_onnx(
        ckpt_path=config["paths"]["checkpoint_destination"], 
        export_dir=config["paths"]["exports_onnx_path"]
    )

if __name__ == "__main__":
    config = load_config()
    export_checkpoint_to_onnx(ckpt_path=config["paths"]["checkpoint_destination"], export_dir=config["paths"]["exports_onnx_path"])