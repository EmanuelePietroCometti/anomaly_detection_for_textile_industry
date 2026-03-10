import os
from anomalib.data import Folder 
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from anomalib.loggers import AnomalibWandbLogger
from dotenv import load_dotenv
import wandb
from huggingface_hub import login
import torch
import warnings
from config import load_config
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def apply_patchcore_model(num_epochs=1, train_batch_size=1, eval_batch_size=1, coreset_sampling_ratio=0.1, backbone="efficientnet_b5", layers=["blocks.4", "blocks.6"]):
    """
    Function to apply the Patchcore model on the textile dataset. It sets up the data, initializes the model, and runs the training and evaluation loop. After testing, it extracts predictions to compute and display a detailed confusion matrix along with accuracy, precision, recall, and F1-score metrics. It also provides warnings about false negatives and false positives.
        Parameters:
        - num_epochs (int): Number of training epochs (default: 1)
        - train_batch_size (int): Batch size for training (default: 1)
        - eval_batch_size (int): Batch size for evaluation (default: 1)
        - coreset_sampling_ratio (float): Coreset sampling ratio for Patchcore (default
        - backbone (str): Backbone for Patchcore (default: "efficientnet_b5")
        - layers (list): List of layers for Patchcore (default: ["blocks.4", "blocks.6"])
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightning.*")
    warnings.filterwarnings("ignore", message=".*persistent_workers=True.*")
    torch.set_float32_matmul_precision('medium')
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
    login(token=hf_token)
    print("Applying Patchcore model...")
    config = load_config("config.yaml")
    dataset_cfg = config.get("paths", {})
    gen_config = config.get("general_configuration", {})
    
    datamodule = Folder(
        name="textiles",
        root=dataset_cfg.get("root", "./data"),
        normal_dir=dataset_cfg.get("train_path", "./data/train").replace("./data/", ""),
        abnormal_dir=dataset_cfg.get("test_reject_path", "./data/test/reject").replace("./data/", ""),
        normal_test_dir=dataset_cfg.get("test_good_path", "./data/test/good").replace("./data/", ""),
        extensions=tuple(gen_config.get("valid_extensions", [".bmp",".BMP"])),
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )

    datamodule.setup()

    logger = AnomalibWandbLogger(project="anomaly-REDA", name="patchcore-run")
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints", 
            filename="patchcore-latest"
        ),
        TimerCallback()
    ]
    print("Initializing Patchcore model...")

    model = Patchcore(
        backbone= backbone,
        layers=layers,
        coreset_sampling_ratio=coreset_sampling_ratio
    )

    engine = Engine(
        max_epochs=num_epochs,          
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",          
    )

    print("Starting training with Patchcore model...")
    engine.fit(model=model, datamodule=datamodule)
    print("Training completed.")

    print("Evaluating Patchcore model...")
    test_results = engine.test(model=model, datamodule=datamodule)
    print("Evaluation completed.")

    #print("Test results:\n", test_results)
    print("\nExtracting predictions for the Confusion Matrix...")
    predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader())

    y_true = []
    y_pred = []

    # Extract the real labels and the ones predicted by the model
    for batch in predictions:
        y_true.extend(batch.gt_label.cpu().numpy())
        y_pred.extend(batch.pred_label.cpu().numpy())

    # 1. Usiamo lo standard: 0=Sano (Negativo), 1=Difetto (Positivo)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # 2. Le metriche ora calcolano correttamente sui Difetti (1) di default
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Estrazione sicura dell'AUROC
    auroc = 0.0
    if isinstance(test_results, list) and len(test_results) > 0:
        auroc = test_results[0].get("auroc", 0.0)
    elif isinstance(test_results, dict):
        auroc = test_results.get("auroc", 0.0)

    # 3. Formatted Print (Dashboard Style) - Logica di Fabbrica Corretta
    print("\n" + "="*65)
    print(" DETAILED REPORT: CONFUSION MATRIX")
    print("="*65)
    print(f"                       | Predicted: GOOD (0) | Predicted: DEFECT (1)")
    print("-" * 65)
    print(f" Actual: GOOD (0)      | [ TN: {tn:<4} ] (Accepted)     | [ FP: {fp:<4} ] (False Alarms)")
    print(f" Actual: DEFECT (1)    | [ FN: {fn:<4} ] (Miss/Escape)  | [ TP: {tp:<4} ] (Found) ")
    print("="*65)
    print(f"    Accuracy    : {acc*100:>6.2f}%")
    print(f"    Precision   : {prec*100:>6.2f}% (When it rejects, it is an actual defect {prec*100:.0f}% of the time)")
    print(f"    Recall      : {rec*100:>6.2f}% (Finds {rec*100:.0f}% of all real defects)")
    print(f"    F1-Score    : {f1:>6.4f}")
    print(f"    AUROC       : {auroc*100:>6.2f}%")
    print("="*65 + "\n")

    if fn > 0:
        print(f"WARNING: The model generated {fn} False Negatives (Missed defects shipped to customer!).")
    if fp > 0:
        print(f"NOTE: The model generated {fp} False Positives (Good parts rejected/scrapped).")