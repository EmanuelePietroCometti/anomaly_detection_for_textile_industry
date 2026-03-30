import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.lightning_model import EfficientAdModelSize
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize

def configure_efficientad(config, custom_weights_path=None):
    """
    Instantiates and configures the EfficientAD model.
    Sets up dynamic transforms, custom weights, and learning rate schedulers.
    """
    ead_cfg = config.get("efficientad_settings", {})
    gen_config = config.get("general_configuration", {})
    
    num_epochs = ead_cfg.get("num_epochs", 100)
    model_size = ead_cfg.get("model_size", "s")
    learning_rate = ead_cfg.get("learning_rate", 1e-4)
    weight_decay = ead_cfg.get("weight_decay", 1e-5)
    img_size = gen_config.get("image_size", [256, 256])

    print(f"Initializing EfficientAD model ({model_size.upper()} size)...")
    if model_size.upper().startswith("S"):
        enum_size = EfficientAdModelSize.S
    else:
        enum_size = EfficientAdModelSize.M
    
    # Model Initialization
    model = EfficientAd(
        model_size=enum_size,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Dynamic Transforms Setup (Attached directly to the model)
    model.transform = Compose([
        Resize(tuple(img_size)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Custom Weights Injection
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
                    
        model.load_state_dict(adapted_state_dict, strict=False)
        print(f"Custom weights injected. Adapted layers: {len(adapted_state_dict)}/{len(state_dict)}")
        if len(adapted_state_dict) == 0:
            print("WARNING: No keys matched. This happens if you pass ResNet weights to a PDN architecture.")
    else:
        print("Using default initialized weights.")

    # Custom Optimizer Override
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
    
    # Override standard Lightning optimizer with the custom one
    model.__class__.configure_optimizers = custom_configure_optimizers

    return model