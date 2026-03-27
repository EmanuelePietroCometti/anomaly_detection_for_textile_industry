from src.dataset_utils import build_mutually_exclusive_datasets
from src.anomaly_patchcore import apply_patchcore_model, export_checkpoint_to_onnx
from src.anomaly_ead import train_efficientad
import argparse
from src.config import load_config
from src.transfer_learning import train_custom_resnet

def main():
    config = load_config()
    argparser = argparse.ArgumentParser(description="Run the Anomaly Detection Pipeline")
    
    argparser.add_argument(
        "--create-dataset",
        default=False,
        action="store_true",
        help="Whether to create the dataset before training"
    )
    argparser.add_argument(
        "--baseline",
        type=str.lower,
        choices=["efficientad", "patchcore"],
        default="efficientad",
        help="Select the model to execute: 'efficientad' or 'patchcore'"
    )
    argparser.add_argument(
        "--run-transfer-learning",
        default=False,
        action="store_true",
        help="Run the backbone (ResNet) fine-tuning before launching the Anomaly Detection model."
    )
    
    args = argparser.parse_args()

    if args.create_dataset:
        print("Starting dataset creation...")
        build_mutually_exclusive_datasets()

    if args.run_transfer_learning:
        print("Starting Transfer Learning...")
        train_custom_resnet(config)
        print("Transfer Learning completed!")

    custom_weights = config["transfer_learning"]["save_path"]

    if args.baseline == "patchcore":
        print("Starting PatchCore training...")
        backbone_name = config["model_architecture"]["backbone"]
        layers = config["model_architecture"]["layers"]
        
        apply_patchcore_model(
            backbone=backbone_name,
            layers=layers,
            custom_weights_path=custom_weights
        )
        print("Patchcore completed")
        export_checkpoint_to_onnx(
            ckpt_path=config["paths"]["checkpoint_destination"], 
            export_dir=config["paths"]["exports_onnx_path"]
        )
        
    elif args.baseline == "efficientad":
        print("Starting EfficientAD training...")
        train_efficientad(
            custom_weights_path=custom_weights
        )
        print("EfficientAD completed!")

if __name__ == "__main__":
    main()