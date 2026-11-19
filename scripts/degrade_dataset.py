"""
===================================================================================
Synthetic Degradation Generator: Low-Light, Poisson-Gaussian Noise, & Motion Blur
===================================================================================
Applies the exact physical degradation recipe described in the research paper:
  1. Low-Light Attenuation: Non-linear gamma curve + global luminance dimming (< 10%)
  2. Signal-Dependent Poisson Shot Noise: Simulates photon starvation
  3. Signal-Independent Gaussian Read Noise: Simulates high-ISO electronic sensor gain (sigma=25)
  4. Linear Motion Blur: Simulates camera shake during slow shutter speed exposures

Usage:
  python scripts/degrade_dataset.py --input_dir dataset/LaPa-Face/normal --output_dir dataset/LaPa-Face/degraded
  python scripts/degrade_dataset.py --input_dir dataset/CASIA-Test/normal --output_dir dataset/CASIA-Test/underexposed_noisy_blurry
===================================================================================
"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def create_motion_blur_kernel(kernel_size=7, angle=45):
    """Generates a directional linear motion blur kernel."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    slope = np.tan(np.deg2rad(angle))

    for x in range(kernel_size):
        offset = x - center
        y = int(np.round(center + offset * slope))
        if 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    if kernel.sum() > 0:
        kernel /= kernel.sum()
    else:
        kernel[center, center] = 1.0
    return kernel


def apply_degradation(image_np, gamma=2.5, dim_factor=0.10, gaussian_sigma=25, blur_ksize=7):
    """
    Applies the compound degradation pipeline to a single RGB uint8 image [0, 255].
    """
    img = image_np.astype(np.float32) / 255.0

    # 1. Optical Motion Blur (simulating slow exposure camera shake)
    kernel = create_motion_blur_kernel(kernel_size=blur_ksize, angle=np.random.choice([0, 30, 45, 90, 135]))
    blurred = cv2.filter2D(img, -1, kernel)

    # 2. Extreme Low-Light Attenuation (Gamma curve + linear luminance drop)
    underexposed = np.power(blurred, gamma) * dim_factor

    # 3. Physical Noise Model:
    # A. Signal-Dependent Poisson Shot Noise (scaled to photon count domain)
    peak = 50.0  # Photon count ceiling for dark scenes
    poisson_noisy = np.random.poisson(np.clip(underexposed, 0, 1) * peak) / peak

    # B. Signal-Independent Gaussian Read Noise (scaled to [0, 1] range)
    sigma_norm = gaussian_sigma / 255.0
    gaussian_noise = np.random.normal(0, sigma_norm, poisson_noisy.shape)

    degraded = poisson_noisy + gaussian_noise

    # Final quantization to 8-bit image domain [0, 255]
    degraded = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
    return degraded


def process_directory(input_dir, output_dir):
    """Walks through files or identity subfolders and generates degraded images."""
    os.makedirs(output_dir, exist_ok=True)
    valid_ext = ('.jpg', '.png', '.jpeg', '.bmp')

    tasks = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(valid_ext):
                src_path = os.path.join(root, f)
                rel_path = os.path.relpath(src_path, input_dir)
                dst_path = os.path.join(output_dir, rel_path)
                tasks.append((src_path, dst_path))

    print(f"Found {len(tasks)} images in {input_dir}. Applying degradation recipe...")

    for src, dst in tqdm(tasks, desc="Synthesizing degradation"):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img = Image.open(src).convert("RGB")
        img_np = np.array(img)

        degraded_np = apply_degradation(img_np)
        Image.fromarray(degraded_np).save(dst)

    print(f"\nAll degraded images successfully saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic Low-Light Noise & Blur Generator")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to clean ground-truth images (e.g. dataset/LaPa-Face/normal)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save degraded output images (e.g. dataset/LaPa-Face/degraded)")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
