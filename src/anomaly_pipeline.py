import torch
import numpy as np
import types
import os
import gc
import shutil
from pathlib import Path
from anomalib.data import Folder
from anomalib.data.utils.split import ValSplitMode
from anomalib.engine import Engine
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from anomalib.metrics import F1AdaptiveThreshold
from src.visualization import save_evaluation_report, plot_auroc_curve
from src.utils import save_prediction_triplet

_IMG_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


def _find_mask(img: Path, candidates: list) -> Path:
    """Maschera di img tra candidates: nome identico, poi <nome>_mask/_gt, poi unica che lo contiene."""
    stem = img.stem
    for rule in (
        lambda m: m.stem == stem,
        lambda m: m.stem in (f"{stem}_mask", f"{stem}_gt"),
        lambda m: stem in m.stem,
    ):
        hits = [m for m in candidates if rule(m)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(f"Maschera ambigua per {img}: {[h.name for h in hits[:5]]}")
    raise FileNotFoundError(f"Nessuna maschera per {img}")


def prepare_test_masks(dataset_path: Path, abnormal_dirs: list, out_dir: Path) -> Path:
    """Costruisce in out_dir una cartella maschere con SOLO le immagini anomale di test.

    anomalib (Folder) abbina maschere e immagini anomale per posizione dopo
    averle ordinate: vuole una maschera per ogni immagine anomala e nient'altro.
    Il ground_truth/ del dataset puo' contenere maschere di tutti gli split
    (e dei good): qui si seleziona solo quello che serve, senza toccare il dataset.
    Ogni maschera prende il nome dell'immagine, cosi' l'ordinamento coincide.
    """
    gt_root = dataset_path / "ground_truth"
    if not gt_root.is_dir():
        raise FileNotFoundError(f"Manca la cartella delle maschere: {gt_root}")

    if out_dir.exists():
        shutil.rmtree(out_dir)

    n = 0
    for rel in abnormal_dirs:                      # es. "test/nodo"
        cls = Path(rel).name
        img_dir = dataset_path / rel
        gt_dir = gt_root / cls
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"Manca la cartella maschere per il difetto '{cls}': {gt_dir}")
        candidates = [m for m in gt_dir.rglob("*") if m.is_file() and m.suffix.lower() in _IMG_EXTS]
        for img in sorted(p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS):
            mask = _find_mask(img, candidates)
            dst = out_dir / cls / img.relative_to(img_dir).with_suffix(mask.suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                dst.symlink_to(mask.resolve())
            except OSError:                        # es. Windows senza permessi per i symlink
                shutil.copy2(mask, dst)
            n += 1

    print(f"[INFO] Maschere di test: {n} selezionate da {gt_root} -> {out_dir}")
    return out_dir


def run_anomaly_pipeline(model, config, project_name="anomaly-pipeline"):
    """
    Unified pipeline to handle data loading, training, and evaluation.
    Utilizes the native Anomalib Folder datamodule for MVTec-structured custom datasets.
    Optimized to prevent memory deadlocks during the inference phase.
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

    # SETUP DATAMODULE USING FOLDER
    root_path = Path(datamodule_cfg.get("root", "./data/mvtec_custom"))
    category = datamodule_cfg.get("category", "textile")
    dataset_path = root_path / category
    
    test_path = dataset_path / "test"
    abnormal_dirs = []
    if test_path.exists():
        abnormal_dirs = [f"test/{d}" for d in os.listdir(test_path) if (test_path / d).is_dir() and d != "good"]

    mask_dir = "ground_truth"
    if abnormal_dirs:
        mask_root = Path(config.get("paths", {}).get("default_root_dir", "results")) / "test_masks" / category
        mask_dir = str(prepare_test_masks(dataset_path, abnormal_dirs, mask_root).resolve())

    datamodule = Folder(
        name=category,
        root=str(dataset_path),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir=abnormal_dirs,
        mask_dir=mask_dir,
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
        num_workers=datamodule_cfg.get("num_workers", 4),
        val_split_mode=ValSplitMode.SAME_AS_TEST
    )

    # ENGINE INITIALIZATION
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.get("paths", {}).get("checkpoint_dir", "checkpoints"),
        filename=f"{model_name}-latest",
        save_top_k=0,
        save_last=True
    )
    
    engine = Engine(
        max_epochs=config.get(f"{model_name.lower()}_configuration", {}).get("num_epochs", 200),
        callbacks=[checkpoint_callback, TimerCallback()],
        accelerator="gpu",
        devices=1,
        default_root_dir=config.get("paths", {}).get("default_root_dir", "results"),
        precision=gen_config.get("precision", "32-true"), 
        log_every_n_steps=50,
        check_val_every_n_epoch=1
    )

    print(f"\n--- Training {model_name} ---")
    if not hasattr(model, 'name') or model.name is None:
        model.name = model_name.lower()
        
    engine.fit(model=model, datamodule=datamodule)

    print(f"\n--- Evaluating {model_name} ---")
    
    # PATCHING FOR LIGHTNING 2.0 COMPATIBILITY
    def patch_callback(cb):
        original = getattr(cb.__class__, "on_test_batch_end")
        def safe_on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0, **kwargs):
            return original(self, trainer, pl_module, (batch if outputs is None else outputs), batch, batch_idx, dataloader_idx, **kwargs)
        cb.on_test_batch_end = types.MethodType(safe_on_test_batch_end, cb)

    for cb in engine.trainer.callbacks:
        if cb.__class__.__name__ in ["PostProcessor", "Evaluator"]:
            patch_callback(cb)
            
    engine.test(model=model, datamodule=datamodule)

    print("\nExtracting predictions for Metrics and AUROC...")
    
    # OPTIMIZED PATCH: CPU offload after internal normalizations
    for cb in engine.trainer.callbacks:
        if cb.__class__.__name__ == "PostProcessor":
            original_predict_batch_end = cb.on_predict_batch_end
            
            def safe_post_process(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
                original_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
                
                target = outputs if outputs is not None else batch
                attributes_to_offload = ["gt_label", "pred_score", "gt_mask", "mask", "anomaly_map", "pred_mask"]
                for attr in attributes_to_offload:
                    val = getattr(target, attr, None)
                    if val is not None and hasattr(val, 'detach'):
                        setattr(target, attr, val.detach().cpu())
                        
            cb.on_predict_batch_end = types.MethodType(safe_post_process, cb)
            break

    # Execute prediction loop safely
    predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader())

    # Only clean cache, do NOT delete the engine so it survives for model export
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # DATA EXTRACTION AND METRICS
    y_true_img, y_scores = [], []
    y_true_masks, y_pred_heatmaps = [], [] 

    print("Processing extracted predictions...")
    for batch in predictions:
        if batch is None: 
            continue
        
        gt = getattr(batch, "gt_label", None)
        score = getattr(batch, "pred_score", None)
        paths = getattr(batch, "image_path", None)
        
        if gt is not None and score is not None:
            y_true_img.extend(gt.numpy().flatten())
            y_scores.extend(score.numpy().flatten())

        mask = getattr(batch, "gt_mask", getattr(batch, "mask", None))
        anomaly_map = getattr(batch, "anomaly_map", None)
        pred_mask = getattr(batch, "pred_mask", None)
        
        if mask is not None:
            y_true_masks.append(mask.numpy())
        if anomaly_map is not None:
            y_pred_heatmaps.append(anomaly_map.numpy())

        if score is not None and paths is not None:
            for i in range(len(score)):
                save_prediction_triplet(
                    img_path=paths[i],
                    score=score[i].item(),
                    anomaly_map=anomaly_map[i].numpy() if anomaly_map is not None else None,
                    pred_mask=pred_mask[i].numpy() if pred_mask is not None else None,
                    config=config,
                    datamodule_cfg=datamodule_cfg
                )

    if len(y_true_img) > 0:
        # Convert lists to tensors
        tensor_scores = torch.tensor(np.array(y_scores)).squeeze() 
        tensor_labels = torch.tensor(np.array(y_true_img)).squeeze()
        tensor_scores = torch.nan_to_num(tensor_scores, nan=0.0, posinf=1.0, neginf=0.0)

        # Ensure spatial arrays are completely flat for threshold computation
        pixel_true_flat = np.concatenate([np.squeeze(m) for m in y_true_masks]).flatten()
        pixel_pred_flat = np.concatenate([np.squeeze(h) for h in y_pred_heatmaps]).flatten()
        
        # Sub-sample pixels to prevent out-of-memory during thresholding (evaluating max 1 Million pixels)
        max_pixels = min(1_000_000, len(pixel_true_flat))
        indices = np.random.choice(len(pixel_true_flat), max_pixels, replace=False)
        tensor_pixel_scores = torch.tensor(pixel_pred_flat[indices])
        tensor_pixel_labels = torch.tensor(pixel_true_flat[indices])
        tensor_pixel_scores = torch.nan_to_num(tensor_pixel_scores, nan=0.0, posinf=1.0, neginf=0.0)

        # Image-Level Threshold Calibration
        img_threshold_calc = F1AdaptiveThreshold(fields=["pred_score", "gt_label"]).to(tensor_scores.device)
        img_threshold_calc.update(types.SimpleNamespace(pred_score=tensor_scores, gt_label=tensor_labels))
        img_thresh = img_threshold_calc.compute().item()

        # Pixel-Level Threshold Calibration
        px_threshold_calc = F1AdaptiveThreshold(fields=["pred_score", "gt_label"]).to(tensor_pixel_scores.device)
        px_threshold_calc.update(types.SimpleNamespace(pred_score=tensor_pixel_scores, gt_label=tensor_pixel_labels))
        px_thresh = px_threshold_calc.compute().item()

        # Update model thresholds dynamically
        model.image_threshold = img_threshold_calc
        model.pixel_threshold = px_threshold_calc
        
        print(f"[CALIBRATION] Image Threshold: {img_thresh:.4f} | Pixel Threshold: {px_thresh:.4f}")

        save_evaluation_report(y_true_masks, y_pred_heatmaps, model_name, backbone, config, img_thresh, px_thresh)
        plot_auroc_curve(y_true_img, tensor_scores.numpy(), model_name, backbone, layers_str, config)
        
    print("[SUCCESS] Pipeline execution and evaluation complete.")
    return engine