from anomalib.models import Supersimplenet

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

    return model