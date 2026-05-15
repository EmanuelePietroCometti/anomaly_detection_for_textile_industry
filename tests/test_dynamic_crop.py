import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import GPUAugmentationCallback

# Ensure the GPUAugmentationCallback class is imported or defined in this script
# from your_module import GPUAugmentationCallback

def test_dynamic_gpu_crop(image_path: str, padding: int = 30):
    """
    Loads an image, simulates the Lightning dataloader tensor format,
    applies the pure-GPU dynamic crop, and plots the comparison.
    """
    # Load the image via OpenCV and convert BGR to RGB for correct plotting
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to PyTorch Tensor [C, H, W] and normalize values to [0.0, 1.0]
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    img_tensor = transform(img_rgb)
    
    # Add Batch dimension to match expected shape: [1, C, H, W]
    img_batch = img_tensor.unsqueeze(0)
    
    # Move tensor to GPU if available to test the pure-GPU logic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_batch = img_batch.to(device)
    
    print(f"Input batch shape: {img_batch.shape}, Device: {img_batch.device}")
    
    # Instantiate your custom callback
    callback = GPUAugmentationCallback(crop_padding=padding)
    
    # Apply the dynamic crop method (passing None for masks in this test)
    cropped_batch, _ = callback._apply_dynamic_gpu_crop(img_batch, masks=None)
    
    print(f"Output batch shape: {cropped_batch.shape}")
    
    # Convert tensors back to NumPy arrays [H, W, C] for Matplotlib visualization
    # Squeeze removes the batch dimension [1, C, H, W] -> [C, H, W]
    # Permute reorders dimensions for image display [C, H, W] -> [H, W, C]
    original_np = img_batch.squeeze(0).cpu().permute(1, 2, 0).numpy()
    cropped_np = cropped_batch.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    # Plot Original vs Cropped side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_np)
    axes[0].set_title("Original Image (Normalized)")
    axes[0].axis("off")
    
    axes[1].imshow(cropped_np)
    axes[1].set_title(f"Dynamic Crop (Padding={padding})")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace with an actual image path from your dataset
    # For instance, an image with dark objects on a white background
    sample_image_path = "data\\dataset_ad\\test\\good\\image_ID 56356_T 30395520.bmp" 
    
    try:
        test_dynamic_gpu_crop(sample_image_path, padding=10)
    except Exception as e:
        print(f"Test failed: {e}")