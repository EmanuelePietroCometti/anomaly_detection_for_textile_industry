import os
import re
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)

def save_evaluation_report(y_true, y_pred, model_name, backbone, config, image_threshold, pixel_threshold):
    """
    Calculates classification metrics (Confusion Matrix, F1, etc.)
    and exports a detailed text report to the disk.
    """
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
        f"    F1-Score    : {f1*100:>6.2f}%\n"
        f"{'='*65}\n"
        f"Image threshold   : {image_threshold:>6.4f} (Performs image-level classification to detect global defects within the input image.)\n"
        f"Pixel threshold   : {pixel_threshold:>6.4f} (Performs pixel-level classification to detect local defects within the input image.)\n"
    )
    print(report_text)

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
    Calculates the False Positive/True Positive rates, plots the AUROC curve,
    and saves the resulting image to the disk.
    """
    print("Generating AUROC Curve...")
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

    # File saving logic
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
