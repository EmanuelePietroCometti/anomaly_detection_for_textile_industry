from src.dataset_utils import build_mutually_exclusive_datasets
from src.anomaly_patchcore import apply_patchcore_model, export_checkpoint_to_onnx
from src.anomaly_ead import train_efficientad
import argparse
from src.config import load_config
from src.transfer_learning import apply_transfer_learning
from src.eda import apply_eda_analysis
from datetime import datetime
import os
import glob

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
        choices=["efficientad", "patchcore", "none"],
        default="none", # CAMBIATO: Messo a 'none' di default così puoi fare solo EDA senza avviare il training per sbaglio
        help="Select the model to execute: 'efficientad' or 'patchcore'"
    )
    argparser.add_argument(
        "--run-transfer-learning",
        default=False,
        action="store_true",
        help="Run the backbone (ResNet) fine-tuning before launching the Anomaly Detection model."
    )
    argparser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Insert manually a timestamp to restart a training process with the same timestamp of a previous run"
    )
    argparser.add_argument(
        "--exploratory-data-analysis",
        default=False,
        action="store_true",
        help="Whether to apply exploratory data analysis before training"
    )

    args = argparser.parse_args()
    
    if args.timestamp is None:
        config['global_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        config["global_timestamp"] = args.timestamp
    
    if args.create_dataset:
        print("Starting dataset creation...")
        build_mutually_exclusive_datasets()

    if args.exploratory_data_analysis:
        print("Starting exploratory data analysis...")
        apply_eda_analysis()

    if args.run_transfer_learning:
        print("Starting Transfer Learning...")
        apply_transfer_learning(config)
        print("Transfer Learning completed!")

    
    if args.baseline in ["patchcore", "efficientad"]:
        custom_weights_dir = config["transfer_learning"]["save_dir"]
        timestamp = config["global_timestamp"]

        search_pattern = os.path.join(custom_weights_dir, f"{timestamp}_*.pth")
        matching_files = glob.glob(search_pattern)

        custom_weights_path = None

        if len(matching_files) == 1:
            custom_weights_path = matching_files[0]
            print(f"[SUCCESS] Custom weights file found: {custom_weights_path}")
        elif len(matching_files) == 0:
            if args.timestamp or args.run_transfer_learning:
                raise FileNotFoundError(f"[ERROR] No weights file found with timestamp {timestamp} in {custom_weights_dir}")
            else:
               
                print(f"[WARNING] No custom weights found for {timestamp}. Make sure the model supports default weights.")
        else:
            raise ValueError(f"[ERROR] Found {len(matching_files)} files with the same timestamp. Cannot disambiguate.")

        if args.baseline == "patchcore":
            print("Starting PatchCore training...")
            backbone_name = config["model_architecture"]["backbone"]
            layers = config["model_architecture"]["layers"]
            
            apply_patchcore_model(
                backbone=backbone_name,
                layers=layers,
                custom_weights_path=custom_weights_path,
                config=config
            )
            print("Patchcore completed")
            export_checkpoint_to_onnx(
                ckpt_path=config["paths"]["checkpoint_destination"], 
                export_dir=config["paths"]["exports_onnx_path"],
                config=config
            )
            
        elif args.baseline == "efficientad":
            print("Starting EfficientAD training...")
            train_efficientad(
                custom_weights_path=custom_weights_path
            )
            print("EfficientAD completed!")

if __name__ == "__main__":
    main()