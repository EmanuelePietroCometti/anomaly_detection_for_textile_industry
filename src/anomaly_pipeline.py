import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.loggers import AnomalibWandbLogger
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from src.visualization import save_evaluation_report, plot_auroc_curve
from src.utils import save_prediction_triplet
import numpy as np

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
    paths = config.get("paths", {})
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

        model.image_threshold.update(preds=tensor_scores, target=tensor_labels)

        img_thresh = model.image_threshold.compute().item()
        px_thresh = img_thresh
        if hasattr(model, "pixel_threshold"):
            try:
                px_thresh = model.pixel_threshold.compute().item()
            except Exception:
                pass

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
