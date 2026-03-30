import os
import torch
from anomalib.models import ReverseDistillation
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize

def configure_rd4ad(config, custom_weights_path=None):
    """
    Instantiates and configures the RD4AD (Reverse Distillation) model.
    Sets up dynamic transforms and custom transfer learning weights if provided.
    """
    gen_config = config.get("general_configuration", {})
    model_arch = config.get("model_architecture", {})
    
    backbone = model_arch.get("backbone", "wide_resnet50_2")
    layers = model_arch.get("layers", ["layer1", "layer2", "layer3"])
    img_size = gen_config.get("image_size", [256, 256])

    print(f"Initializing RD4AD model with backbone {backbone}...")

    # Model Initialization
    model = ReverseDistillation(
        backbone=backbone,
        layers=layers,
    )

    # Dynamic Transforms Setup (Attached directly to the model)
    model.transform = Compose([
        Resize(tuple(img_size)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Custom Weights Injection (Applied to the Teacher Encoder)
    if custom_weights_path and os.path.exists(custom_weights_path):
        print(f"Loading custom weights from: {custom_weights_path}")
        state_dict = torch.load(custom_weights_path, map_location="cpu")
        
        anomalib_state_dict = model.state_dict()
        adapted_state_dict = {}
        
        # Dynamic key mapping to align standard ResNet weights with RD4AD structure
        for custom_key, tensor in state_dict.items():
            for anomalib_key in anomalib_state_dict.keys():
                if custom_key in anomalib_key:
                    adapted_state_dict[anomalib_key] = tensor
                    break
                    
        model.load_state_dict(adapted_state_dict, strict=False)
        print(f"Custom weights injected. Adapted layers: {len(adapted_state_dict)}/{len(state_dict)}")
    else:
        print("Using default ImageNet weights for RD4AD Teacher Encoder.")

    return model