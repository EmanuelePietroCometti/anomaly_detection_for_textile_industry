# src/anomaly_ead.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.lightning_model import EfficientAdModelSize
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
# from src.metrics import TargetRecallThreshold 

def configure_efficientad(config):
    """
    Instantiates and configures the EfficientAD model.
    Sets up dynamic transforms, learning rate schedulers, and adaptive thresholding metrics.
    Custom weights injection is intentionally omitted as PDN architecture is incompatible 
    with standard ResNet transfer learning.
    """
    ead_cfg = config.get("efficientad_configuration", {})
    gen_config = config.get("general_configuration", {})
    
    num_epochs = ead_cfg.get("num_epochs", 100)
    model_size = ead_cfg.get("model_size", "s")
    learning_rate = ead_cfg.get("learning_rate", 1e-4)
    weight_decay = ead_cfg.get("weight_decay", 1e-5)
    img_size = gen_config.get("image_size", [256, 256])
    imagenete_dir = ead_cfg.get("imagenette_dir", "./data/imagenette_for_efficientad")

    print(f"Initializing EfficientAD model ({model_size.upper()} size)...")
    if model_size.upper().startswith("S"):
        enum_size = EfficientAdModelSize.S
    else:
        enum_size = EfficientAdModelSize.M
    
    # Model Initialization
    model = EfficientAd(
        imagenet_dir=imagenete_dir,
        model_size=enum_size,
        lr=learning_rate,
        weight_decay=weight_decay,
        padding=True,
        pad_maps=False
    )

    # model.image_threshold = TargetRecallThreshold(target_recall=0.99)
    # model.pixel_threshold = TargetRecallThreshold(target_recall=0.99)

    # Dynamic Transforms Setup
    model.transform = Compose([
        Resize(tuple(img_size)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Using default initialized weights (PDN architecture).")

    # Custom Optimizer Override
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

    return model