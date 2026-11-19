"""
===================================================================================
Stage 1: Baseline Evaluation (Base Paper - Low-FaceNet, IEEE TIM 2024)
===================================================================================
Evaluates Low-FaceNet on the DEGRADED CASIA-Test benchmark:
  - Input: dataset/CASIA-Test/underexposed_noisy_blurry (Severely degraded: low-light, high-ISO Poisson-Gaussian noise, motion blur)
  - Output: dataset/CASIA-Test/SOTAs/LowFaceNet_Baseline

Expected Output: ~12.6% closed-set identification accuracy with frequent RetinaFace
landmark detection dropouts ([!] No face detected) caused by Retinex noise amplification (1/I^2).
===================================================================================
"""

import os
import sys
import shutil
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# Make sure we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import EnhanceNet


def find_all_degraded_images(root_dir):
    """Walk through identity subdirectories and collect all degraded image paths."""
    tasks = []
    valid_ext = ('.jpg', '.png', '.jpeg', '.bmp')

    for person_id in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person_id)
        if not os.path.isdir(person_path):
            continue
        for fname in os.listdir(person_path):
            if fname.lower().endswith(valid_ext):
                tasks.append((person_id, fname))
    return tasks


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Load Base Paper Low-FaceNet Weights ---
    weight_candidates = [
        "checkpoint/Low-FaceNet.pth",
        "checkpoint/best_casia_weight.pth",
        "checkpoint/casia_weight.pth"
    ]
    weight_path = next((p for p in weight_candidates if os.path.isfile(p)), None)
    if not weight_path:
        print("Error: Could not find Low-FaceNet weights in checkpoint/ directory.")
        return

    print(f"Loading Low-FaceNet model weights from: {weight_path}")
    net = EnhanceNet().to(device)
    net.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    net.eval()

    # --- 2. Ensure face_module weights exist ---
    os.makedirs("face_module/model_data", exist_ok=True)
    for w in ["facenet_mobilenet.pth", "Retinaface_mobilenet0.25.pth"]:
        sw = os.path.join("checkpoint", w)
        dw = os.path.join("face_module/model_data", w)
        if os.path.exists(sw) and not os.path.exists(dw):
            shutil.copy2(sw, dw)

    # --- 3. Reset evaluation caches ---
    for cache_dir in ("encoding", "database"):
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)

    # --- 4. Prepare Paths (DEGRADED CASIA-TEST INPUT) ---
    degraded_input_root = "dataset/CASIA-Test/underexposed_noisy_blurry"
    output_root = "dataset/CASIA-Test/SOTAs/LowFaceNet_Baseline"
    os.makedirs(output_root, exist_ok=True)

    image_list = find_all_degraded_images(degraded_input_root)
    print(f"Found {len(image_list)} DEGRADED images across all 100 identity folders in {degraded_input_root}.")

    to_tensor = T.ToTensor()

    # --- 5. Enhance Degraded Images with Low-FaceNet ---
    print("\nProcessing DEGRADED images through Low-FaceNet front-end...")
    with torch.no_grad():
        for person_id, fname in tqdm(image_list, desc="Low-FaceNet Baseline (Degraded Input)"):
            src = os.path.join(degraded_input_root, person_id, fname)
            dst_dir = os.path.join(output_root, person_id)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fname)

            img = Image.open(src).convert("RGB")
            tensor_in = to_tensor(img).unsqueeze(0).to(device)

            result = net(tensor_in)
            if isinstance(result, tuple):
                result = result[0]
            result = torch.clamp(result, 0.0, 1.0)

            T.ToPILImage()(result.squeeze(0).cpu()).save(dst)

    print(f"\nAll enhanced baseline images saved to: {output_root}")

    # --- 6. Subprocess Evaluation on Degraded CASIA-Test ---
    print("\n" + "="*70)
    print("EVALUATING STAGE 1: Low-FaceNet Alone on DEGRADED CASIA-Test Benchmark")
    print("="*70)
    os.system("python eval.py --dataset_name=CASIA-Test --method_name=LowFaceNet_Baseline")


if __name__ == "__main__":
    main()
