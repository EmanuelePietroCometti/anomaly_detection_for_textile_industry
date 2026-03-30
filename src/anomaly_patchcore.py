import os
import torch
from anomalib.models import Patchcore
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize, CenterCrop
from torchvision.transforms import InterpolationMode
from metrics import TargetRecallThreshold


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

    # Dynamic Transforms Setup
    if "efficientnet" in backbone:
        model.transform = Compose([
            Resize(crop_size, interpolation=InterpolationMode.BICUBIC), 
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        model.transform = Compose([
            Resize(image_size),
            CenterCrop(crop_size),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Apply Custom Adaptive Thresholds
    model.image_threshold = TargetRecallThreshold(target_recall=0.99)
    model.pixel_threshold = TargetRecallThreshold(target_recall=0.99)

    # Custom Weights Injection
    if custom_weights_path and os.path.exists(custom_weights_path):
        print(f"Loading custom weights from: {custom_weights_path}")
        state_dict = torch.load(custom_weights_path, map_location="cpu")
        
        anomalib_state_dict = model.state_dict()
        adapted_state_dict = {}
        
        for custom_key, tensor in state_dict.items():
            for anomalib_key in anomalib_state_dict.keys():
                if custom_key in anomalib_key:
                    adapted_state_dict[anomalib_key] = tensor
                    break
                    
        model.load_state_dict(adapted_state_dict, strict=False)
        print(f"Custom weights injected. Adapted layers: {len(adapted_state_dict)}/{len(state_dict)}")
    else:
        print("Using default ImageNet weights for backbone.")

    return model