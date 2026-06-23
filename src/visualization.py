import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import label
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc, 
    average_precision_score, roc_auc_score
)

def _normalize_data(data) -> np.ndarray:
    """
    Safely converts PyTorch Lightning prediction outputs into a uniform 3D NumPy array (N, H, W).
    Crucially removes redundant single-channel dimensions (N, 1, H, W) to prevent metric evaluation failures.
    """
    # Base conversion for single tensor/array
    if hasattr(data, 'cpu'):
        data = data.cpu().detach().numpy()
        
    if isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, list):
        # Convert all elements safely
        clean_list = [x.cpu().detach().numpy() if hasattr(x, 'cpu') else np.array(x) for x in data]
        
        if not clean_list:
            return np.array([])
            
        first_elem = clean_list[0]
        
        if first_elem.ndim == 0 and first_elem.dtype == object:
            raise TypeError("[DATA FORMAT ERROR] The input contains dictionaries instead of raw tensors.")
        
        # Concatenate batches
        if first_elem.ndim >= 3:
            arr = np.concatenate(clean_list, axis=0)
        else:
            arr = np.asarray(clean_list)
    else:
        arr = np.array(data)

    # Robust channel squeezing: converts (N, 1, H, W) or (N, H, W, 1) into (N, H, W)
    if arr.ndim == 4:
        if arr.shape[1] == 1:
            arr = np.squeeze(arr, axis=1)
        elif arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
            
    return arr

def check_inputs_diagnostic(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Prints diagnostic information about evaluation shapes, dimensions, and data types.
    """
    print(f"\n{'='*20} INPUT DIAGNOSTIC {'='*20}")
    print(f"[INFO] y_true (Ground Truth) -> Shape: {y_true.shape} | Dtype: {y_true.dtype} | Dimensions: {y_true.ndim}")
    print(f"[INFO] y_pred (Predicted Map) -> Shape: {y_pred.shape} | Dtype: {y_pred.dtype} | Dimensions: {y_pred.ndim}")
    
    # Assertions to prevent AUPRO crashes
    assert y_true.ndim == 3, f"[ERROR] y_true must be 3D (N, H, W), found {y_true.ndim}D. Squeeze logic failed."
    assert y_pred.ndim == 3, f"[ERROR] y_pred must be 3D (N, H, W), found {y_pred.ndim}D. Squeeze logic failed."
    assert y_true.shape == y_pred.shape, "[ERROR] Shape mismatch between y_true and y_pred"
    print(f"{'='*58}\n")

def compute_ap_loc(y_true_mask: np.ndarray, y_pred_heatmap: np.ndarray) -> float:
    """
    Computes Average Precision for localization (AP-loc) at the pixel level.
    """
    y_true_flat = y_true_mask.flatten()
    y_pred_flat = y_pred_heatmap.flatten()
    return average_precision_score(y_true_flat, y_pred_flat)

def compute_aupro(y_true_mask: np.ndarray, y_pred_heatmap: np.ndarray, max_fpr: float = 0.3) -> float:
    """
    Computes the Area Under the Per-Region Overlap (AUPRO) curve.
    Includes linear interpolation at the max_fpr boundary for mathematically flawless AUC integration.
    """
    y_true_mask = y_true_mask.astype(np.uint8)
    
    labeled_anomalies = np.zeros_like(y_true_mask, dtype=int)
    current_label = 0
    
    # Label connected components per image
    for i in range(y_true_mask.shape[0]):
        labels, num = label(y_true_mask[i], return_num=True)
        if num > 0:
            labels[labels > 0] += current_label
            current_label += num
        labeled_anomalies[i] = labels
        
    num_anomalies = current_label
    if num_anomalies == 0:
        return 0.0 
        
    # FAST EXTRACTION
    region_preds = []
    for i in range(y_true_mask.shape[0]):
        labels_in_slice = np.unique(labeled_anomalies[i])
        for lab in labels_in_slice:
            if lab > 0: 
                region_preds.append(y_pred_heatmap[i][labeled_anomalies[i] == lab])
                
    neg_preds = y_pred_heatmap[y_true_mask == 0]
    total_negative_pixels = neg_preds.size
    
    # Nota: 100 soglie sono un'approssimazione ottima per la velocità. 
    # Per una precisione scientifica assoluta, potresti voler aumentare questo numero (es. 200 o 500).
    thresholds = np.linspace(y_pred_heatmap.min(), y_pred_heatmap.max(), 100)
    pro_scores, fprs = [], []
    
    for th in thresholds:
        pro_sum = sum((preds >= th).sum() / preds.size for preds in region_preds)
        pro_scores.append(pro_sum / num_anomalies)
        
        fp_pixels = (neg_preds >= th).sum()
        fprs.append(fp_pixels / total_negative_pixels)
        
    # Ordinamento ascendente necessario per l'AUC
    fprs = np.array(fprs)
    pro_scores = np.array(pro_scores)
    sort_idx = np.argsort(fprs)
    fprs, pro_scores = fprs[sort_idx], pro_scores[sort_idx]
    
    # Estrazione dei punti validi e calcolo interpolazione
    valid_mask = fprs <= max_fpr
    
    if not valid_mask.any():
        return 0.0
        
    fprs_valid = fprs[valid_mask].tolist()
    pro_valid = pro_scores[valid_mask].tolist()

    # Interpolazione Lineare esatta al bordo max_fpr
    exceeded_idx = np.where(~valid_mask)[0]
    if len(exceeded_idx) > 0 and len(fprs_valid) > 0:
        idx_next = exceeded_idx[0]
        fpr_next = fprs[idx_next]
        pro_next = pro_scores[idx_next]
        
        fpr_prev = fprs_valid[-1]
        pro_prev = pro_valid[-1]
        
        # Calcolo del valore PRO esatto corrispondente a max_fpr (es. 0.3)
        if fpr_next > fpr_prev: # Prevenzione divisione per zero
            slope = (pro_next - pro_prev) / (fpr_next - fpr_prev)
            pro_interp = pro_prev + slope * (max_fpr - fpr_prev)
            
            fprs_valid.append(max_fpr)
            pro_valid.append(pro_interp)

    # Per un AUC corretto, la curva deve idealmente partire da FPR = 0
    if fprs_valid[0] > 0.0:
        fprs_valid.insert(0, 0.0)
        pro_valid.insert(0, pro_valid[0])

    fprs_valid = np.array(fprs_valid)
    pro_valid = np.array(pro_valid)
    
    # Normalizzazione finale dell'asse X a [0, 1]
    fprs_normalized = fprs_valid / max_fpr
    
    return auc(fprs_normalized, pro_valid)

def compute_image_level_auroc(y_true_masks: np.ndarray, y_pred_heatmaps: np.ndarray) -> float:
    """
    Computes the Image-level AUROC for anomaly detection via Max Projection.
    """
    y_true_image = np.array([np.max(mask) for mask in y_true_masks])
    y_pred_image = np.array([np.max(heatmap) for heatmap in y_pred_heatmaps])
    
    if len(np.unique(y_true_image)) < 2:
        raise ValueError("Image-level AUROC requires at least one normal and one anomalous image in the batch.")
        
    return roc_auc_score(y_true_image, y_pred_image)

def save_evaluation_report(y_true, y_pred, model_name, backbone, config, image_threshold, pixel_threshold):
    """
    Calculates classification metrics, runs diagnostics, and exports a text report.
    """
    # Normalize and strip extra channels
    y_true = _normalize_data(y_true)
    y_pred = _normalize_data(y_pred)
    
    # Run Diagnostic to ensure 3D (N, H, W) compliance
    check_inputs_diagnostic(y_true, y_pred)

    # Compute Metrics
    image_auroc = compute_image_level_auroc(y_true_masks=y_true, y_pred_heatmaps=y_pred)
    
    y_true_img = np.array([np.max(mask) for mask in y_true])
    y_pred_img_scores = np.array([np.max(heatmap) for heatmap in y_pred])
    y_pred_img_binary = (y_pred_img_scores >= image_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_img, y_pred_img_binary, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true_img, y_pred_img_binary)
    prec = precision_score(y_true_img, y_pred_img_binary, zero_division=0)
    rec = recall_score(y_true_img, y_pred_img_binary, zero_division=0)
    f1 = f1_score(y_true_img, y_pred_img_binary, zero_division=0)
    
    ap = compute_ap_loc(y_true_mask=y_true, y_pred_heatmap=y_pred)
    aupro = compute_aupro(y_true_mask=y_true, y_pred_heatmap=y_pred)

    # Generate Report
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
        f"    F1-Score    : {f1*100:>6.2f}%\n"
        f"    AUROC       : {image_auroc*100:>6.2f}% (Image-level)\n"
        f"    AUPRO       : {aupro*100:>6.2f}% (Localization)\n"
        f"    AP-loc      : {ap*100:>6.2f}% (Localization)\n"
        f"{'='*65}\n"
        f"Image threshold   : {image_threshold:>6.4f}\n"
        f"Pixel threshold   : {pixel_threshold:>6.4f}\n"
    )
    print(report_text)

    # Save Report
    paths = config.get("paths", {})
    timestamp = config.get("global_timestamp", "latest")

    model_arch = config.get("model_architecture", {})
    layers = model_arch.get("layers", ["custom_layers"])
    layers_str = "_".join(layers) if isinstance(layers, list) else str(layers)
    layers_str = re.sub(r'[\\/*?:"<>|{}\']', "", layers_str)

    report_filename = f"{timestamp}_evaluation_report_{backbone}_{layers_str}.txt"
    report_dir = paths.get("report_path", f"results/{model_name}/report")
    os.makedirs(report_dir, exist_ok=True)
    report_filepath = os.path.join(report_dir, report_filename)

    with open(report_filepath, "w", encoding="utf-8") as file:
        file.write(report_text)

    print(f"[SUCCESS] Report successfully saved to: {report_filepath}")

def plot_auroc_curve(y_true, y_scores, model_name, backbone, layers_str, config):
    """
    Plots the global False Positive/True Positive ROC curve.
    """
    print("Generating AUROC Curve...")
    
    y_true = _normalize_data(y_true)
    y_scores = _normalize_data(y_scores)
        
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} ROC (AUC = {roc_auc:.4f})')
    plt.fill_between(fpr, tpr, alpha=0.15, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)

    plt.title(
        f'Receiver Operating Characteristic - {model_name}\n'
        f'Architecture Details: {backbone} | {layers_str}\n',
        fontsize=10
    )
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)

    paths = config.get("paths", {})
    timestamp = config.get("global_timestamp", "latest")
    safe_layers_str = re.sub(r'[\\/*?:"<>|{}\']', "", layers_str)

    results_dir = Path(paths.get("auroc_path", f"results/{model_name}/auroc"))
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{timestamp}_{model_name}_auroc_{backbone}_{safe_layers_str}.png"
    plot_path = results_dir / filename

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] AUROC Curve exported successfully to: {plot_path}")