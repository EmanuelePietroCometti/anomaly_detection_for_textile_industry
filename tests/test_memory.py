import torch
import torchvision.models as models

try:
    from torchinfo import summary
except ImportError:
    print("Please install torchinfo: pip install torchinfo")

def profile_wide_resnet(batch_size: int = 32, image_size: int = 224, precision_bytes: int = 2):
    """
    Profiles the exact VRAM footprint for wide_resnet50_2.
    Resolves the HalfTensor vs FloatTensor mismatch by aligning the model's 
    weights to the input precision and passing 'input_data' directly.
    """
    print(f"[INFO] Profiling wide_resnet50_2 | Batch: {batch_size} | Res: {image_size}x{image_size} | Precision: {precision_bytes} bytes")
    
    model = models.wide_resnet50_2()
    
    if precision_bytes == 2:
        model = model.half()  # Convert model weights to FP16
        dtype = torch.float16
    else:
        dtype = torch.float32
        
    input_shape = (batch_size, 3, image_size, image_size)
    dummy_input = torch.randn(input_shape, dtype=dtype)
    
    # Use torchinfo with 'input_data' instead of 'input_size' to avoid UserWarnings
    model_stats = summary(
        model, 
        input_data=dummy_input, 
        verbose=0
    )
    
    params_mb = model_stats.to_megabytes(model_stats.total_param_bytes)
    activations_mb = model_stats.to_megabytes(model_stats.total_output_bytes)
    
    # Calculate the structural VRAM for training
    # Weights + Gradients + AdamW states
    # Note: AdamW states are usually maintained in FP32 (8 bytes total per parameter) 
    # even when the model is in FP16 for numerical stability.
    adam_multiplier = 8 / precision_bytes
    structural_mb = params_mb + params_mb + (params_mb * adam_multiplier) 
    
    # Total estimation (Adding 800MB for base CUDA overhead)
    total_training_vram_mb = structural_mb + activations_mb + 800 
    
    print("\n--- ESTIMATED VRAM FOR TRAINING ---")
    print(f"Structural (Model + Grads + AdamW): ~{structural_mb:.2f} MB")
    print(f"Activations (Forward + Backward):   ~{activations_mb:.2f} MB")
    print(f"CUDA Context Overhead:              ~800.00 MB")
    print("-" * 40)
    print(f"TOTAL EXPECTED VRAM PEAK:           ~{total_training_vram_mb / 1024:.2f} GB")

if __name__ == "__main__":
    profile_wide_resnet(batch_size=4, image_size=512, precision_bytes=2)