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

def train_efficientad(custom_weights_path=None):
    """
    Function to train the EfficientAD model on the textile dataset. 
    It reads parameters dynamically from config.yaml and supports custom weights injection.
    
    Args:
        custom_weights_path (str, optional): Path to a .pth file containing pre-trained weights.
                                             Note: EfficientAD uses a custom PDN architecture, 
                                             so standard ResNet weights will be mostly ignored.
    """
    # Optimization for RTX A4000 Tensor Cores
    torch.set_float32_matmul_precision('medium')
    
    # Load credentials
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    if hf_token:
        login(token=hf_token)
    
    # Read Configuration
    config = load_config("config.yaml")
    datamodule_cfg = config.get("datamodule_configruation", {})
    gen_config = config.get("general_configuration", {})
    ead_cfg = config.get("efficientad_settings", {})
    
    num_epochs = ead_cfg.get("num_epochs", 100)
    train_batch_size = ead_cfg.get("train_batch_size", 8)
    eval_batch_size = ead_cfg.get("eval_batch_size", 8)
    model_size = ead_cfg.get("model_size", "s")
    learning_rate = ead_cfg.get("learning_rate", 1e-4)
    weight_decay = ead_cfg.get("weight_decay", 1e-5)
    num_workers = ead_cfg.get("num_workers", 4)
    img_size = gen_config.get("image_size", [256, 256])
    
    print("Setting up Dataset for EfficientAD...")
    transform = Compose([
        Resize(tuple(img_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pre_processor = PreProcessor(transform=transform)
    
    datamodule = Folder(
        name="textiles_ead",
        root=datamodule_cfg.get("root", "./data"),
        normal_dir=datamodule_cfg.get("train_path", "./data/train").replace("./data/", ""),
        abnormal_dir=datamodule_cfg.get("test_reject_path", "./data/test/reject").replace("./data/", ""),
        normal_test_dir=datamodule_cfg.get("test_good_path", "./data/test/good").replace("./data/", ""),
        extensions=tuple(gen_config.get("valid_extensions", [".bmp",".BMP"])),
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        pre_process=pre_processor
    )
    datamodule.setup()
    
    logger = AnomalibWandbLogger(project="anomaly-REDA", name="efficientad-run")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="efficientad-latest",
        monitor="val_loss"
    )
    timer_callback = TimerCallback()

    print(f"Initializing EfficientAD model ({model_size.upper()} size)...")
    if model_size.upper().startswith("S"):
        enum_size = EfficientAdModelSize.S
    else:
        enum_size = EfficientAdModelSize.M
        
    model = EfficientAd(
        model_size = enum_size,
        lr = learning_rate,
        weight_decay = weight_decay,
    )

    # --- WEIGHTS INJECTION ---
    if custom_weights_path and os.path.exists(custom_weights_path):
        print(f"Loading custom weights from: {custom_weights_path}")
        state_dict = torch.load(custom_weights_path, map_location="cpu")
        
        anomalib_state_dict = model.state_dict()
        adapted_state_dict = {}
        
        # Dynamic key mapping
        for custom_key, tensor in state_dict.items():
            for anomalib_key in anomalib_state_dict.keys():
                if custom_key in anomalib_key:
                    adapted_state_dict[anomalib_key] = tensor
                    break
                    
        missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
        print(f"Custom weights injected. Adapted layers: {len(adapted_state_dict)}/{len(state_dict)}")
        if len(adapted_state_dict) == 0:
            print("WARNING: No keys matched. This happens if you pass ResNet weights to a PDN architecture.")
    else:
        print("Using default initialized weights.")

    def custom_configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        # Cosine Annealing scheduler scales learning rate over epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", 
                "frequency": 1
            }
        }
    
    # Override standard optimizer with the custom one
    model.__class__.configure_optimizers = custom_configure_optimizers

    engine = Engine(
        max_epochs=num_epochs, 
        logger=logger,
        callbacks=[checkpoint_callback, timer_callback] ,
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

    # Extract real labels and predicted labels
    for batch in predictions:
        y_true.extend(batch.gt_label.cpu().numpy())
        y_pred.extend(batch.pred_label.cpu().numpy())

    # Calculate metrics
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
    print(f" Predicted: GOOD (0)        | [ TP: {tp:<4} ]    | [ FP: {fp:<4} ]")
    print(f" Predicted: DEFECT (1)      | [ FN: {fn:<4} ]    | [ TN: {tn:<4} ]")
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