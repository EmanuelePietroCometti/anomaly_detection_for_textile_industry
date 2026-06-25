import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort
import os

def preprocess_image(image_path: str, input_shape: tuple) -> np.ndarray:
    """
    Reads and preprocesses the image to match the ONNX model requirements.
    
    Args:
        image_path: Path to the input image.
        input_shape: Expected shape from the ONNX session (batch, channels, height, width).
    
    Returns:
        A preprocessed numpy array of shape (1, C, H, W).
    """
    _, channels, h, w = input_shape
    target_size = (w, h) if isinstance(w, int) and isinstance(h, int) else (256, 256)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img

def benchmark_inference(session: ort.InferenceSession, input_name: str, input_data: np.ndarray, iterations: int):
    """
    Runs the inference loop to calculate average processing time.
    """
    print("Running warm-up (This might take longer for TensorRT if engine is building)...")
    for _ in range(50):
        session.run(None, {input_name: input_data})

    print(f"Running benchmark for {iterations} iterations...")
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        session.run(None, {input_name: input_data})
        
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_batch_time = total_time / iterations
    
    return total_time, avg_batch_time

def main():
    parser = argparse.ArgumentParser(description="ONNX Inference Benchmark: Batch 1 vs Batch 17")
    parser.add_argument("--model", type=str, required=True, help="Path to the .onnx model file")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image to replicate")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of inference iterations (default: 1000)")
    # Added 'tensorrt' to the choices
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "tensorrt"], help="Execution provider")
    args = parser.parse_args()

    # Configure the Execution Provider
    providers = []
    
    if args.device == "tensorrt":
        # TensorRT configuration with engine caching to avoid rebuilding the engine every run
        trt_cache_path = "./trt_engines"
        os.makedirs(trt_cache_path, exist_ok=True)
        
        trt_options = {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': trt_cache_path,
            'trt_fp16_enable': True, # Enable Mixed Precision if your GPU supports it
        }
        
        # Fallback cascade: TensorRT -> CUDA -> CPU
        providers = [
            ('TensorrtExecutionProvider', trt_options),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
    elif args.device == "cuda":
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    # Initialize ONNX Session
    try:
        session = ort.InferenceSession(args.model, providers=providers)
        
        # Verify which provider was actually assigned
        active_providers = session.get_providers()
        print(f"[INFO] ONNX Model loaded successfully.")
        print(f"[INFO] Requested device: {args.device.upper()}")
        print(f"[INFO] Active Execution Providers: {active_providers}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model: {e}")
        return

    # Extract input metadata
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    print(f"[INFO] Expected model input shape: {input_shape}")

    # Prepare Single Image (Batch Size 1)
    img_b1 = preprocess_image(args.image, input_shape)

    print("\n--- Starting Batch Size 1 Benchmark ---")
    _, avg_time_b1 = benchmark_inference(session, input_name, img_b1, args.iterations)
    avg_time_per_img_b1 = avg_time_b1 / 1.0

    # Auto-detect if the model is Patchcore
    model_filename = os.path.basename(args.model).lower()
    is_patchcore = "patchcore" in model_filename

    if is_patchcore:
        print("\n--- Skipping Batch Size 17 Benchmark ---")
        print("[INFO] Detected a Patchcore model. ONNX Patchcore models are restricted to Batch Size = 1.")
        
        print("\n===========================================")
        print("            BENCHMARK RESULTS              ")
        print("===========================================")
        print(f"Model Type: PATCHCORE")
        print(f"Iterations per configuration: {args.iterations}")
        print(f"Provider Priority: {providers[0] if isinstance(providers[0], str) else providers[0][0]}")
        print("-" * 43)
        print(f"BATCH SIZE 1:")
        print(f"  Avg time per image:  {avg_time_per_img_b1 * 1000:.2f} ms")
        print("===========================================")
    
    else:
        img_b17 = np.repeat(img_b1, 17, axis=0)
        
        print("\n--- Starting Batch Size 17 Benchmark ---")
        try:
            _, avg_time_b17 = benchmark_inference(session, input_name, img_b17, args.iterations)
            avg_time_per_img_b17 = avg_time_b17 / 17.0
        except Exception as e:
            print(f"\n[ERROR] Batch 17 failed. Error: {e}")
            return

        print("\n===========================================")
        print("            BENCHMARK RESULTS              ")
        print("===========================================")
        print(f"Iterations per configuration: {args.iterations}")
        print(f"Provider Priority: {providers[0] if isinstance(providers[0], str) else providers[0][0]}")
        print("-" * 43)
        print(f"BATCH SIZE 1:")
        print(f"  Avg time per batch:  {avg_time_b1 * 1000:.2f} ms")
        print(f"  Avg time per image:  {avg_time_per_img_b1 * 1000:.2f} ms")
        print("-" * 43)
        print(f"BATCH SIZE 17:")
        print(f"  Avg time per batch:  {avg_time_b17 * 1000:.2f} ms")
        print(f"  Avg time per image:  {avg_time_per_img_b17 * 1000:.2f} ms")
        print("===========================================")
        
        speedup = avg_time_per_img_b1 / avg_time_per_img_b17
        print(f"CONCLUSION: Batch 17 processes an individual image {speedup:.2f}x faster than Batch 1.")

if __name__ == "__main__":
    main()