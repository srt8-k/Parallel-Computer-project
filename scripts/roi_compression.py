"""
ROI-Aware Compression for Medical Images
-----------------------------------------
Automatically detects the diagnostically significant region (ROI) using
Otsu's thresholding, then applies K-Means with:
  - High K (more colors) inside the ROI  → preserve diagnostic detail
  - Low  K (fewer colors) in background  → aggressive compression
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


# ── ROI Detection ────────────────────────────────────────────────────────────

def otsu_threshold(gray_array):
    """Pure-numpy Otsu threshold — no extra dependencies needed."""
    histogram, bin_edges = np.histogram(gray_array.flatten(), bins=256, range=(0, 256))
    total = gray_array.size

    best_threshold = 0
    best_variance = 0.0
    cumulative_sum = 0
    cumulative_weight = 0

    for t in range(256):
        cumulative_weight += histogram[t]
        cumulative_sum += t * histogram[t]

        if cumulative_weight == 0 or cumulative_weight == total:
            continue

        weight_bg = cumulative_weight / total
        weight_fg = 1.0 - weight_bg
        mean_bg = cumulative_sum / cumulative_weight
        mean_fg = (gray_array.sum() - cumulative_sum) / (total - cumulative_weight)

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = t

    return best_threshold


def detect_roi(image_path):
    """
    Returns a boolean mask (same H×W as the image) where True = ROI.
    Uses Otsu's method on the grayscale version of the image.
    """
    image = Image.open(image_path).convert('L')
    gray = np.array(image, dtype=np.uint8)

    threshold = otsu_threshold(gray)
    roi_mask = gray > threshold

    print(f"Otsu threshold: {threshold}")
    print(f"ROI pixels:        {roi_mask.sum():,} ({roi_mask.mean()*100:.1f}% of image)")
    print(f"Background pixels: {(~roi_mask).sum():,}")

    return roi_mask, threshold


# ── K-Means helpers ──────────────────────────────────────────────────────────

def _kmeans_pixels(pixels, k, iterations=10):
    """Run K-Means on a (N, 3) float32 pixel array. Returns quantized pixels."""
    import random
    indices = np.random.choice(len(pixels), size=min(k, len(pixels)), replace=False)
    centroids = pixels[indices].copy()

    for _ in range(iterations):
        distances = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = []
        for i in range(k):
            cluster = pixels[labels == i]
            if len(cluster) > 0:
                new_centroids.append(cluster.mean(axis=0))
            else:
                new_centroids.append(pixels[random.randint(0, len(pixels) - 1)])
        new_centroids = np.array(new_centroids)

        if np.all(np.abs(centroids - new_centroids) < 0.00001):
            break
        centroids = new_centroids

    return np.array([centroids[l] for l in labels])


# ── ROI Compression ──────────────────────────────────────────────────────────

def roi_kmeans_compression(image_path, k_roi=16, k_background=4,
                           iterations=10, savePath=None, show_result=True):
    """
    Apply K-Means compression with different quality levels:
      - ROI (diagnostically significant area): k_roi colors
      - Background:                            k_background colors

    Parameters
    ----------
    image_path   : path to input medical image
    k_roi        : number of colors for the ROI region (higher = better quality)
    k_background : number of colors for the background (lower = more compression)
    """
    if not os.path.exists(image_path):
        print("Image does not exist")
        return None

    print(f"\n{'='*60}")
    print("ROI-Aware K-Means Compression")
    print(f"  ROI colors:        {k_roi}")
    print(f"  Background colors: {k_background}")
    print(f"{'='*60}")

    # Load and detect ROI
    image = Image.open(image_path).convert('RGB')
    roi_mask, threshold = detect_roi(image_path)

    # resize mask to match image dimensions if needed
    if roi_mask.shape != (image.size[1], image.size[0]):
        roi_mask = np.array(
            Image.fromarray(roi_mask.astype(np.uint8) * 255).resize(image.size, Image.NEAREST)
        ) > 127

    data = np.array(image, dtype=np.float32) / 255.0
    height, width, _ = data.shape
    pixels = data.reshape(-1, 3)
    flat_mask = roi_mask.flatten()

    # Compress ROI region
    roi_pixels = pixels[flat_mask]
    print(f"\nCompressing ROI with K={k_roi}...")
    if len(roi_pixels) >= k_roi:
        compressed_roi = _kmeans_pixels(roi_pixels, k_roi, iterations)
    else:
        compressed_roi = roi_pixels.copy()

    # Compress background region
    bg_pixels = pixels[~flat_mask]
    print(f"Compressing background with K={k_background}...")
    if len(bg_pixels) >= k_background:
        compressed_bg = _kmeans_pixels(bg_pixels, k_background, iterations)
    else:
        compressed_bg = bg_pixels.copy()

    # Merge back
    result_pixels = np.zeros_like(pixels)
    result_pixels[flat_mask] = compressed_roi
    result_pixels[~flat_mask] = compressed_bg

    result_image_array = (result_pixels.reshape(height, width, 3) * 255).astype(np.uint8)
    result_image = Image.fromarray(result_image_array)

    # Compression ratio
    originalBits = height * width * 24
    roi_count = flat_mask.sum()
    bg_count = (~flat_mask).sum()
    compressedBits = (k_roi * 24 + roi_count * np.log2(k_roi)) + \
                     (k_background * 24 + bg_count * np.log2(k_background))
    ratio = originalBits / compressedBits

    print(f"\nCompression ratio: {ratio:.4f}")

    if savePath:
        result_image.save(savePath)
        print(f"Saved to {savePath}")

    if show_result:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(roi_mask, cmap='hot')
        axes[1].set_title(f"Detected ROI\n(Otsu threshold = {threshold})")
        axes[1].axis('off')

        axes[2].imshow(result_image)
        axes[2].set_title(f"ROI Compression\nROI: K={k_roi} | BG: K={k_background}")
        axes[2].axis('off')

        plt.suptitle("ROI-Aware K-Means Compression for Medical Images", fontsize=13)
        plt.tight_layout()
        plt.show()

    return result_image, ratio, roi_mask


def calculate_psnr_roi(original_path, compressed_image):
    import math
    original = np.array(Image.open(original_path).convert('RGB'), dtype=np.float32)
    compressed = np.array(compressed_image, dtype=np.float32)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim_roi(original_path, compressed_image):
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("scikit-image not installed. Run: pip install scikit-image")
        return None

    original = np.array(Image.open(original_path).convert('RGB'), dtype=np.float32)
    compressed = np.array(compressed_image, dtype=np.float32)
    return ssim(original, compressed, data_range=255.0, channel_axis=2)
