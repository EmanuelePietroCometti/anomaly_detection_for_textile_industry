from anomalib.models import Supersimplenet
import torch
import types

def custom_configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

def configure_supersimplenet(config):
    """"
    Configure the SuperSimpleNet model based on the provided configuration.
    """

    model_arch = config.get("model_architecture", {})
    supersimplenet_config = config.get("supersimplenet_configuration", {})
    backbone = model_arch.get("backbone", "resnet18")
    layers = model_arch.get("layers", ["layer1", "layer2", "layer3"])

    print(f"\n[INFO] Configuring SuperSimpleNet with backbone: {backbone} and layers: {layers}")
    
    backbone = backbone+".tv_in1k"

    model = Supersimplenet(
        perlin_threshold=supersimplenet_config.get("perlin_threshold", 0.2),
        backbone=backbone, 
        layers=layers,
        supervised=supersimplenet_config.get("supervised", False)
    )

    custom_lr = supersimplenet_config.get("learning_rate")
    custom_weight_decay = supersimplenet_config.get("weight_decay")
    custom_num_epochs = supersimplenet_config.get("num_epochs")
    if custom_lr is not None:
        print(f"[INFO] Injecting custom learning rate: {custom_lr} into SuperSimpleNet")
        model.learning_rate = custom_lr
        model.weight_decay = custom_weight_decay
        model.configure_optimizers = types.MethodType(custom_configure_optimizers, model)

    return model