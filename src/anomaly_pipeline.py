import torch
from anomalib.data import Folder
from anomalib.metrics import F1AdaptiveThreshold
from anomalib.engine import Engine
from anomalib.loggers import AnomalibWandbLogger
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from src.visualization import save_evaluation_report, plot_auroc_curve
from src.utils import save_prediction_triplet
import types
import numpy as np
from anomalib.metrics.threshold import ManualThreshold
from anomalib.data.utils.split import ValSplitMode, TestSplitMode

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
    layers_str = "_".join(layers) if isinstance(layers, list) else str(layers)

    model_specific_cfg = config.get(f"{model_name.lower()}_configuration", {})
    
    train_bs = model_specific_cfg.get("train_batch_size", datamodule_cfg.get("train_batch_size", 32))
    eval_bs = model_specific_cfg.get("eval_batch_size", datamodule_cfg.get("eval_batch_size", 32))
    
    mask_dir = datamodule_cfg.get("mask_dir", None)
    if mask_dir is not None:
        mask_dir = mask_dir.replace("./data/", "")

    datamodule = Folder(
        name="textiles_dataset",
        root=datamodule_cfg.get("root", "./data"),
        normal_dir=datamodule_cfg.get("train_dir", "train/good").replace("./data/", ""),
        abnormal_dir=datamodule_cfg.get("test_dir_reject", "test/reject").replace("./data/", ""),
        normal_test_dir=datamodule_cfg.get("test_dir_good", "test/good").replace("./data/", ""),
        mask_dir=mask_dir,
        extensions=tuple(gen_config.get("valid_extensions", [".bmp", ".BMP"])),
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
        num_workers=datamodule_cfg.get("num_workers", 4),
        val_split_mode=ValSplitMode.SAME_AS_TEST
    )
    datamodule.setup()

    logger = AnomalibWandbLogger(project=project_name, name=model_name)
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename=f"{model_name}-latest")

    max_epochs = config.get(f"{model_name.lower()}_configuration", {}).get("num_epochs", 1)

    engine = Engine(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, TimerCallback()],
        accelerator="gpu",
        devices=1,
        precision=gen_config.get("precision", "16-mixed")
    )

    print(f"\n--- Training {model_name} ---")
    engine.fit(model=model, datamodule=datamodule)

    if gen_config.get("flag_threshold", False):
        manual_threshold_value = gen_config.get("manual_threshold", None)
        if manual_threshold_value is not None:
            print(f"\n[INFO] Overriding dynamic threshold with manual value: {manual_threshold_value}")
            model.image_threshold = ManualThreshold(default_value=manual_threshold_value)
            model.pixel_threshold = ManualThreshold(default_value=manual_threshold_value)

    print(f"\n--- Evaluating {model_name} ---")
    engine.test(model=model, datamodule=datamodule)

    print("\nExtracting predictions for Metrics and AUROC...")
    predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader())

    y_true, y_pred, y_scores = [], [], []

    for batch in predictions:
        if isinstance(batch, dict):
            gt = batch.get("gt_label", batch.get("label", []))
            pred = batch.get("pred_label", [])
            score = batch.get("pred_score", [])
            image_paths = batch.get("image_path", [])
            anomaly_maps = batch.get("anomaly_map", [])
            pred_masks = batch.get("pred_mask", [])
        else:
            gt = getattr(batch, "gt_label", getattr(batch, "label", []))
            pred = getattr(batch, "pred_label", [])
            score = getattr(batch, "pred_score", [])
            image_paths = getattr(batch, "image_path", [])
            anomaly_maps = getattr(batch, "anomaly_map", [])
            pred_masks = getattr(batch, "pred_mask", [])

        if len(score) > 0 and len(image_paths) > 0:
            for idx, single_score_tensor in enumerate(score):
                single_score = single_score_tensor.item() if hasattr(single_score_tensor, 'item') else single_score_tensor
                img_path = image_paths[idx]

                a_map = anomaly_maps[idx] if anomaly_maps is not None and len(anomaly_maps) > idx else None
                p_mask = pred_masks[idx] if pred_masks is not None and len(pred_masks) > idx else None

                save_prediction_triplet(
                    img_path=img_path,
                    score=single_score,
                    anomaly_map=a_map,
                    pred_mask=p_mask,
                    config=config,
                    datamodule_cfg=datamodule_cfg
                )
        if len(gt) > 0:
            y_true.extend(gt.cpu().numpy() if hasattr(gt, 'cpu') else gt)
        if len(pred) > 0:
            y_pred.extend(pred.cpu().numpy() if hasattr(pred, 'cpu') else pred)
        if len(score) > 0:
            y_scores.extend(score.cpu().numpy() if hasattr(score, 'cpu') else score)

    if len(y_true) > 0:
        tensor_scores = torch.tensor(np.array(y_scores)).squeeze() 
        tensor_labels = torch.tensor(np.array(y_true)).squeeze()

        if not hasattr(model, "image_threshold") or model.image_threshold is None:
            print("[WARNING] 'image_threshold' missing in model. Initializing F1AdaptiveThreshold dynamically.")
            
            model.image_threshold = F1AdaptiveThreshold(fields=["pred_score", "gt_label"])
            model.image_threshold = model.image_threshold.to(tensor_scores.device)

        mock_batch = types.SimpleNamespace(
            pred_score=tensor_scores,
            gt_label=tensor_labels
        )
        
        model.image_threshold.update(mock_batch)
        
        img_thresh = model.image_threshold.compute().item()
  
        px_thresh = img_thresh
        if hasattr(model, "pixel_threshold") and model.pixel_threshold is not None:
            try:
                px_thresh = model.pixel_threshold.compute().item()
            except Exception as e:
                print(f"[WARNING] Skipping pixel_threshold computation: {e}")
        else:
            print("[INFO] 'pixel_threshold' not found. Defaulting to image_threshold value.")

        print(f"[DEBUG] Calculated Thresholds -> Image: {img_thresh:.4f}, Pixel: {px_thresh:.4f}")

        save_evaluation_report(
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_name,
            backbone=backbone,
            config=config,
            image_threshold=img_thresh,
            pixel_threshold=px_thresh
        )
        plot_auroc_curve(y_true, y_scores, model_name, backbone, layers_str, config)
    else:
        print("[WARNING] Ground truth labels were not found in predict step. Skipping plots.")
    return engine
