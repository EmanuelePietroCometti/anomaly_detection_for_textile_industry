import os
import torch
import wandb
from dotenv import load_dotenv
from huggingface_hub import login
from anomalib.data import Folder 
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from anomalib.loggers import AnomalibWandbLogger
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from config import load_config
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from anomalib.models.image.efficient_ad.lightning_model import EfficientAdModelSize
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from anomalib.pre_processing import PreProcessor

def train_efficientad(num_epochs=100, train_batch_size=8, eval_batch_size=8, model_size="s", learning_rate=1e-4, weight_decay=1e-5, num_workers=4):
    """
    Function to train the EfficientAD model on the textile dataset. It sets up the data, initializes the model, and runs the training and evaluation loop. After testing, it extracts predictions to compute and display a detailed confusion matrix along with accuracy, precision, recall, and F1-score metrics. It also provides warnings about false negatives and false positives.
        Parameters:
        - num_epochs (int): Number of training epochs (default: 100)
        - train_batch_size (int): Batch size for training (default: 8)
        - eval_batch_size (int): Batch size for evaluation (default: 8)
        - model_size (str): Size of the EfficientAD model, either "s" for small or "m" for medium (default: "s")
        - learning_rate (float): Learning rate for the optimizer (default: 1e-4)
        - weight_decay (float): Weight decay for the optimizer (default: 1e-
    """
    # Optimization for RTX A4000 Tensor Cores
    torch.set_float32_matmul_precision('medium')
    
    # Credentials
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    if hf_token:
        login(token=hf_token)
    
    # Read Configuration
    config = load_config("config.yaml")
    dataset_cfg = config.get("paths", {})
    gen_config = config.get("general_configuration", {})
    
    print("Setting up Dataset for EfficientAD...")
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pre_processor = PreProcessor(transform=transform)
    datamodule = Folder(
        name="textiles_ead",
        root=dataset_cfg.get("root", "./data"),
        normal_dir=dataset_cfg.get("train_path", "./data/train").replace("./data/", ""),
        abnormal_dir=dataset_cfg.get("test_reject_path", "./data/test/reject").replace("./data/", ""),
        normal_test_dir=dataset_cfg.get("test_good_path", "./data/test/good").replace("./data/", ""),
        extensions=tuple(gen_config.get("valid_extensions", [".bmp",".BMP"])),
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=4,
        pre_process=pre_processor
    )
    datamodule.setup()
    
    logger = AnomalibWandbLogger(project="anomaly-REDA", name="efficientad-run")
    
    callbacks = [
        ModelCheckpoint(dirpath="checkpoints", filename="efficientad-latest"),
        TimerCallback()
    ]

    
    print(f"Initializing EfficientAD model ({model_size} size)...")
    if model_size.upper().startswith("S"):
        enum_size = EfficientAdModelSize.S
    else:
        enum_size = EfficientAdModelSize.M
    model = EfficientAd(
        model_size = enum_size,
        lr = learning_rate,
        weight_decay = weight_decay,

    )

    def custom_configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
    
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", 
                "frequency": 1
            }
        }
    
    model.__class__.configure_optimizers = custom_configure_optimizers

    # Set 100 epochs to allow the Student network to learn the textures properly
    engine = Engine(
        max_epochs=num_epochs, 
        logger=logger,
        callbacks=callbacks,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        precision=16 if torch.cuda.is_available() else 32,
        check_val_every_n_epoch=5,
    )

    print("Starting Training... (The loss will decrease as it learns your normality)")
    engine.fit(model=model, datamodule=datamodule)

    print("Evaluating the model...")
    engine.test(model=model, datamodule=datamodule)
    print("Evaluation completed.")
    print("\nExtracting predictions for the Confusion Matrix...")
    predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader())

    y_true = []
    y_pred = []

    # Extract the real labels and the ones predicted by the model
    for batch in predictions:
        y_true.extend(batch.gt_label.cpu().numpy())
        y_pred.extend(batch.pred_label.cpu().numpy())

    # Calculate Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "="*65)
    print(" DETAILED REPORT: CONFUSION MATRIX")
    print("="*65)
    print(f"                            | True: GOOD (0)               | True: DEFECT (1)")
    print("-" * 65)
    print(f" Predicted: GOOD (0)        | [ TN: {tp:<4} ] (Accepted)     | [ FP: {fp:<4} ] (False Alarms)")
    print(f" Predicted: DEFECT (1)      | [ FN: {fn:<4} ] (Miss/Escape)  | [ TP: {tn:<4} ] (Found) ")
    print("="*65)
    print(f"    Accuracy    : {acc*100:>6.2f}%")
    print(f"    Precision   : {prec*100:>6.2f}% (When it rejects, it is an actual defect {prec*100:.0f}% of the time)")
    print(f"    Recall      : {rec*100:>6.2f}% (Finds {rec*100:.0f}% of all real defects)")
    print(f"    F1-Score    : {f1:>6.4f}")
    print("="*65 + "\n")

    if fn > 0:
        print(f"WARNING: The model generated {fn} False Negatives (Missed defects shipped to customer!).")
    if fp > 0:
        print(f"NOTE: The model generated {fp} False Positives (Good parts rejected).")

if __name__ == "__main__":
    train_efficientad()