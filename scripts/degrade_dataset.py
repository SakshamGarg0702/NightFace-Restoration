import os
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def create_motion_blur_kernel(kernel_size=7, angle=45):
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
    img = image_np.astype(np.float32) / 255.0

    # Motion blur
    kernel = create_motion_blur_kernel(kernel_size=blur_ksize, angle=np.random.choice([0, 30, 45, 90, 135]))
    blurred = cv2.filter2D(img, -1, kernel)

    # Low-light attenuation
    underexposed = np.power(blurred, gamma) * dim_factor

    # Poisson shot noise + Gaussian read noise
    peak = 50.0
    poisson_noisy = np.random.poisson(np.clip(underexposed, 0, 1) * peak) / peak

    sigma_norm = gaussian_sigma / 255.0
    gaussian_noise = np.random.normal(0, sigma_norm, poisson_noisy.shape)

    degraded = poisson_noisy + gaussian_noise
    degraded = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
    return degraded


def process_directory(input_dir, output_dir):
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

    print(f"Processing {len(tasks)} images from {input_dir}...")

    for src, dst in tqdm(tasks, desc="Degrading images"):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img = Image.open(src).convert("RGB")
        img_np = np.array(img)

        degraded_np = apply_degradation(img_np)
        Image.fromarray(degraded_np).save(dst)

    print(f"Saved degraded images to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic low-light degradations")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
