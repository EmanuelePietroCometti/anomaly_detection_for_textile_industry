import os
import torch
import types
from anomalib.models import ReverseDistillation
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
from anomalib.models.image.reverse_distillation.anomaly_map import AnomalyMapGenerationMode

def custom_configure_optimizers(self):
    """
    Bound method to dynamically configure the optimizer for PyTorch Lightning.
    Accesses hyperparameters directly from the 'self' instance.
    Uses list comprehension instead of lambda to preserve pickling/export compatibility.
    """
    # Extract only parameters that require gradients (Student and Bottleneck)
    trainable_parameters = [p for p in self.parameters() if p.requires_grad]
    
    opt_type = self.optimizer_type.lower()
    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(trainable_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
    elif opt_type == "sgd":
        optimizer = torch.optim.SGD(trainable_parameters, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
    else: # Default to Adam
        optimizer = torch.optim.Adam(trainable_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        
    return optimizer


def configure_rd4ad(config):
    """
    Instantiates and configures the RD4AD (Reverse Distillation) model.
    Sets up dynamic transforms, extracts hyperparameters from config, 
    and injects a custom optimizer setup avoiding pickling issues.
    """
    gen_config = config.get("general_configuration", {})
    model_arch = config.get("model_architecture", {})
    rd4ad_config = config.get("reversedistillation_configuration", {})
    img_size = gen_config.get("image_size", [256, 256])

    # Extract hyperparameters from the config file (with safe defaults)
    num_epochs = rd4ad_config.get("num_epochs", 200)
    learning_rate = rd4ad_config.get("learning_rate", 0.005)
    weight_decay = rd4ad_config.get("weight_decay", 1e-4)
    optimizer_type = rd4ad_config.get("optimizer", "adam")

    backbone = model_arch.get("backbone", "wide_resnet50_2")
    layers = model_arch.get("layers", ["layer1", "layer2", "layer3"])

    print(f"Initializing RD4AD model with backbone {backbone}...")
    print(f"[INFO] RD4AD Params -> Epochs: {num_epochs} | LR: {learning_rate} | WD: {weight_decay} | Opt: {optimizer_type}")

    # Model Initialization
    model = ReverseDistillation(
        backbone=backbone,
        layers=layers,
        anomaly_map_mode=AnomalyMapGenerationMode.MULTIPLY
    )

    # Attach hyperparameters to the model instance for the custom optimizer to access
    model.learning_rate = learning_rate
    model.weight_decay = weight_decay
    model.optimizer_type = optimizer_type
    model.num_epochs = num_epochs

    # Properly bind the function as a method of the model instance
    model.configure_optimizers = types.MethodType(custom_configure_optimizers, model)

    # Dynamic Transforms Setup
    model.transform = Compose([
        Resize(tuple(img_size)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return model