import os
import cv2
import logging

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["KMP_WARNINGS"] = "0"

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
cv2.setNumThreads(0)

import glob
import argparse
from datetime import datetime
from src.config import load_config
from src.dataset_utils import build_mutually_exclusive_datasets
from src.transfer_learning import apply_transfer_learning
from src.eda import apply_eda_analysis
from src.anomaly_pipeline import run_anomaly_pipeline
from src.anomaly_patchcore import configure_patchcore
from src.anomaly_ead import configure_efficientad
from src.anomaly_rd4ad import configure_rd4ad
from src.anomaly_supersimplenet import configure_supersimplenet
from src.utils import export_model_to_onnx, rename_run_and_update_symlink, save_config_file, export_model_to_pt

def main():
    config = load_config()
    paths = config["paths"]
    model_arch=config["model_architecture"]
    
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
        choices=["efficientad", "patchcore", "rd4ad", "supersimplenet", "none"],
        default="none",
        help="Select the model to execute: 'efficientad', 'patchcore', or 'rd4ad'"
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
    argparser.add_argument(
        "--mode",
        type=str.lower,
        choices=["unsupervised", "supervised"],
        default="unsupervised",
        help="Select the training mode for the anomaly detection model: 'unsupervised' or 'supervised'"
    )
    argparser.add_argument(
        "--retrain",
        default=False,
        action="store_true",
        help="Trigger supervised retraining on a small set of misclassified samples (only applicable if --mode is set to 'supervised')"
    )

    args = argparser.parse_args()
    
    if args.retrain:
        print("\n[WARNING] Supervised retraining enabled. Make sure to set --mode to 'supervised' and have the necessary labeled data available for retraining.")

        args.mode = "supervised"

        config["datamodule_configuration"]["root"] = "./data/dataset_retraining"
        config["datamodule_configuration"]["train_dir"] =  "train/good"
        config["datamodule_configuration"]["abnormal_dir"] = "train/reject"
        config["datamodule_configuration"]["train_batch_size"] = 2

        config["datamodule_configuration"]["mask_dir"] = "train/reject"
    
    if args.mode == "supervised":
        print("\n[INFO] Supervised training mode selected. Enabling supervised configuration for applicable models.")
        config["supersimplenet_configuration"]["supervised"] = True

    if args.timestamp is None:
        config['global_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        config["global_timestamp"] = args.timestamp
    
    if args.create_dataset:
        print("\nStarting dataset creation...")
        build_mutually_exclusive_datasets()

    if args.exploratory_data_analysis:
        print("\nStarting exploratory data analysis...")
        apply_eda_analysis()

    if args.run_transfer_learning:
        print("\nStarting Transfer Learning...")
        apply_transfer_learning(config)
        print("Transfer Learning completed!")
    
    if args.baseline in ["patchcore", "efficientad", "rd4ad", "supersimplenet"]:
        custom_weights_dir = config["transfer_learning"]["save_dir"]
        timestamp = config["global_timestamp"]

        search_pattern = os.path.join(custom_weights_dir, f"{timestamp}_*.pth")
        matching_files = glob.glob(search_pattern)

        custom_weights_path = None

        if len(matching_files) == 1:
            custom_weights_path = matching_files[0]
            print(f"\n[SUCCESS] Custom weights file found: {custom_weights_path}")
        elif len(matching_files) == 0:
            if args.timestamp or args.run_transfer_learning:
                print(f"\n[WARNING] No weights file found with timestamp {timestamp} in {custom_weights_dir}")
            else:
                print(f"\n[INFO] No custom weights found for {timestamp}. Make sure the model supports default weights.")
        else:
            raise ValueError(f"\n[ERROR] Found {len(matching_files)} files with the same timestamp. Cannot disambiguate.")

        if args.baseline == "efficientad":
            print("[INFO] EfficientAD selected. Disabling incompatible custom weights injection (PDN architecture).")
            custom_weights_path = None
            
        elif args.baseline == "rd4ad":
            if custom_weights_path:
                print(f"[INFO] RD4AD selected. Enabling custom weights injection from: {custom_weights_path}")
            else:
                print("[INFO] RD4AD selected but no custom weights found. Proceeding with robust ImageNet weights.")
        elif args.baseline == "supersimplenet":
            print("[INFO] SuperSimpleNet selected. Disabling incompatible custom weights injection (ResNet backbone).")
            custom_weights_path = None

        print(f"\nConfiguring {args.baseline.upper()}...")
        if args.baseline == "patchcore":
            model = configure_patchcore(config, custom_weights_path)
            
        elif args.baseline == "efficientad":
            model = configure_efficientad(config)
            
        elif args.baseline == "rd4ad":
            model = configure_rd4ad(config)

        elif args.baseline == "supersimplenet":
            model = configure_supersimplenet(config)

        print(f"\nStarting unified training/evaluation pipeline for {args.baseline.upper()}...")
        engine = run_anomaly_pipeline(model, config) 
        print(f"\n[SUCCESS] Entire pipeline for {args.baseline.upper()} completed successfully!")

        
        rename_run_and_update_symlink(symlink_path=paths["symlink_path"], backbone=model_arch["backbone"], layers=model_arch["layers"], config=config)
        save_config_file(config=config, model=model)

        print(f"\nStarting ONNX Export for {args.baseline.upper()}...")
        export_model_to_onnx(model=model, config=config, engine=engine)
        

        print(f"\nStarting PyTorch (.pt) Export for {args.baseline.upper()}...")
        export_model_to_pt(model=model, config=config, engine=engine)

        print("\n[SUCCESS] Execution finished!")

if __name__ == "__main__":
    main()