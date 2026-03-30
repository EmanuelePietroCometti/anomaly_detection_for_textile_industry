import torch
from anomalib.metrics import F1AdaptiveThreshold

class TargetRecallThreshold(F1AdaptiveThreshold):
    """
    Custom Adaptive Threshold that guarantees a minimum target Recall 
    while maximizing Precision to minimize false positives (scraps).
    """
    def __init__(self, target_recall=0.99, default_value=0.5, **kwargs):
        fields = kwargs.pop("fields", ["pred_score", "gt_label"])
        super().__init__(fields=fields, **kwargs)
        self.target_recall = target_recall

    def compute(self) -> torch.Tensor:
        # Compute Precision, Recall, and all possible Thresholds from the PR curve
        precision, recall, thresholds = self.precision_recall_curve.compute()
        
        # Find indices where the model achieves the requested target Recall
        valid_indices = torch.where(recall >= self.target_recall)[0]
        
        if len(valid_indices) > 0:
            # Filter precision and thresholds using only the safe indices
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