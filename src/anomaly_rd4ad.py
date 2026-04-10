import os
import torch
from anomalib.models import ReverseDistillation
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
# from src.metrics import TargetRecallThreshold

def configure_rd4ad(config):
    """
    Instantiates and configures the RD4AD (Reverse Distillation) model.
    Sets up dynamic transforms and safely injects custom transfer learning 
    weights strictly into the Teacher encoder.
    """
    gen_config = config.get("general_configuration", {})
    model_arch = config.get("model_architecture", {})
    
    # RD4AD paper officially recommends Wide-ResNet50, but supports others
    backbone = model_arch.get("backbone", "wide_resnet50_2")
    layers = model_arch.get("layers", ["layer1", "layer2", "layer3"])
    img_size = gen_config.get("image_size", [256, 256])

    print(f"Initializing RD4AD model with backbone {backbone}...")

    # Model Initialization
    model = ReverseDistillation(
        backbone=backbone,
        layers=layers,
    )

    # Dynamic Transforms Setup
    model.transform = Compose([
        Resize(tuple(img_size)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # model.image_threshold = TargetRecallThreshold(target_recall=0.99)
    # model.pixel_threshold = TargetRecallThreshold(target_recall=0.99)

    return model