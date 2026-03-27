from src.dataset_utils import build_mutually_exclusive_datasets
from src.anomaly_patchcore import apply_patchcore_model, export_checkpoint_to_onnx
from src.anomaly_ead import train_efficientad
import argparse
from src.config import load_config
from src.transfer_learning import apply_transfer_learning
from datetime import datetime
import os
import glob

def main():
    config = load_config()
    config['global_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        apply_transfer_learning(config)
        print("Transfer Learning completed!")

    custom_weights_dir = config["transfer_learning"]["save_dir"]
    timestamp = config["global_timestamp"]

    search_pattern = os.path.join(custom_weights_dir, f"{timestamp}_*.pth")
    matching_files = glob.glob(search_pattern)

    if len(matching_files) == 1:
        custom_weights_path = matching_files[0]
        print(f"[SUCCESS] Custom weights file found: {custom_weights_path}")
    elif len(matching_files) == 0:
        raise FileNotFoundError(f"[ERROR] No weights file found with timestamp {timestamp} in {custom_weights_dir}")
    else:
        raise ValueError(f"[ERROR] Found {len(matching_files)} files with the same timestamp. Cannot disambiguate.")


    if args.baseline == "patchcore":
        print("Starting PatchCore training...")
        backbone_name = config["model_architecture"]["backbone"]
        layers = config["model_architecture"]["layers"]
        
        apply_patchcore_model(
            backbone=backbone_name,
            layers=layers,
            custom_weights_path=custom_weights_path
        )
        print("Patchcore completed")
        export_checkpoint_to_onnx(
            ckpt_path=config["paths"]["checkpoint_destination"], 
            export_dir=config["paths"]["exports_onnx_path"]
        )
        
    elif args.baseline == "efficientad":
        print("Starting EfficientAD training...")
        train_efficientad(
            custom_weights_path=custom_weights_path
        )
        print("EfficientAD completed!")

if __name__ == "__main__":
    main()