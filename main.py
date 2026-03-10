from src.dataset_utils import create_dataset
from anomaly_patchcore import apply_patchcore_model
from anomaly_ead import train_efficientad
import argparse

def main():
    argparser = argparse.ArgumentParser(description="Run the Anomaly Detection Pipeline")
    argparser.add_argument(
        "--create-dataset", 
        default=False ,
        action="store_true", 
        help="Whether to create the dataset before training"
        )
    argparser.add_argument(
        "--baseline", 
        type=str.lower,
        choices=["efficientad", "patchcore"], 
        default="efficientad",
        help="Select the modol to execute: 'efficientad' or 'patchcore'"
    )
    argparser.add_argument(
        "--num-epochs", 
        type=int, 
        default=100, 
        help="Numbers of epochs (ex. 100 for EAD, 1 for Patchcore)"
    )
    argparser.add_argument(
        "--train-batch-size", 
        type=int, 
        default=8, 
        help="Training batch size"
    )
    argparser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8, 
        help="Test batch size"
    )
    argparser.add_argument(
        "--model-size", 
        type=str.lower,
        choices=["s", "m"],
        default="s", 
        help="Student-teacher model size for EfficientAD (S or M). Ignored if baseline is Patchcore."
    )
    argparser.add_argument(
        "--coreset-sampling-ratio", 
        type=float, 
        default=0.1, 
        help="Coreset sampling ratio for Patchcore (ex. 0.1 means 10% of patches). Ignored if baseline is EfficientAD."
    )
    argparser.add_argument(
        "--backbone", 
        type=str.lower, 
        default="efficientnet_b5", 
        help="Backbone for Patchcore (ex. 'efficientnet_b5'). Ignored if baseline is EfficientAD."
    )
    argparser.add_argument(
        "--layers", 
        type=str, 
        default="blocks.4,blocks.6", 
        help="Comma-separated layers for Patchcore (ex. 'blocks.4,blocks.6'). Ignored if baseline is EfficientAD."
    )
    argparser.add_argument(
        "--learning-rate", 
        type=float,
        default=1e-4,
        help="Learning rate for EfficientAD training. Ignored if baseline is Patchcore."
    )
    argparser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for EfficientAD training. Ignored if baseline is Patchcore."
    )
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading. Ignored if baseline is Patchcore."
    )
    args = argparser.parse_args()

    if args.create_dataset:
        create_dataset()
    
    if args.baseline == "patchcore":
        apply_patchcore_model(args.num_epochs, args.train_batch_size, args.eval_batch_size, args.coreset_sampling_ratio, backbone=args.backbone, layers=args.layers.split(","))
    elif args.baseline == "efficientad":
        train_efficientad(args.num_epochs, args.train_batch_size, args.eval_batch_size, args.model_size, args.learning_rate, args.weight_decay, args.num_workers)

if __name__ == "__main__":
    main()