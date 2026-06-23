import os
import torch
from anomalib.models import Patchcore
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize, CenterCrop
from torchvision.transforms import InterpolationMode
# from src.metrics import TargetRecallThreshold


def configure_patchcore(config, custom_weights_path=None):
    """
    Instantiates and configures the Patchcore model, adapting transforms
    and injecting custom transfer learning weights if provided.
    """
    patchcore_cfg = config.get("patchcore_configuration", {})
    gen_config = config.get("general_configuration", {})
    model_arch = config.get("model_architecture", {})
    
    backbone = model_arch.get("backbone", "resnet18")
    layers = model_arch.get("layers", ["layer2", "layer3"])
    image_size = gen_config.get("image_size", [256, 256])
    crop_size = gen_config.get("crop_size", [224, 224])

    # Model Initialization
    model = Patchcore(
        backbone=backbone,
        layers=layers,
        coreset_sampling_ratio=patchcore_cfg.get("coreset_sampling_ratio", 0.1),
        num_neighbors=patchcore_cfg.get("num_nearest_neighbors", 9),
    )

    return model