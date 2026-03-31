import torch
from torchmetrics import Metric
from torchmetrics.classification import BinaryPrecisionRecallCurve

class TargetRecallThreshold(Metric):
    """
    Custom Adaptive Threshold that guarantees a minimum target Recall
    while maximizing Precision to minimize false positives (scraps).
    """
    full_state_update: bool = False

    def __init__(self, target_recall: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.target_recall = target_recall

        self.precision_recall_curve = BinaryPrecisionRecallCurve()
        self.add_state("value", default=torch.tensor(0.5), dist_reduce_fx="mean")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Accumulates batch predictions and ground truth labels.
        """
        self.precision_recall_curve.update(preds, target)

    def compute(self) -> torch.Tensor:
        """
        Calculates the optimal threshold based on accumulated data.
        """
        # Compute Precision, Recall, and all possible Thresholds
        precision, recall, thresholds = self.precision_recall_curve.compute()

        precision = precision[:-1]
        recall = recall[:-1]

        # Find indices where the model achieves the requested target Recall
        valid_indices = torch.where(recall >= self.target_recall)[0]

        if len(valid_indices) > 0:
            # Filter precision and thresholds using only the valid, safe indices
            valid_precisions = precision[valid_indices]
            valid_thresholds = thresholds[valid_indices]

            # Pick the index with the highest Precision to minimize scraps
            best_idx = torch.argmax(valid_precisions)
            self.value = valid_thresholds[best_idx]
        else:
            # Fallback: If the target is mathematically unreachable,
            # pick the threshold that yields the absolute maximum Recall possible.
            print(f"\n[WARNING] Target recall {self.target_recall} unreachable. Defaulting to max possible recall.")
            best_idx = torch.argmax(recall)
            self.value = thresholds[best_idx]

        return self.value
