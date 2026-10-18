import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import time
import os


def initialize_centroids(pixels, k):
    indices = np.random.choice(len(pixels), size=k, replace=False)
    return pixels[indices]


def assign_centroids(centroids, pixels):
    # vectorized: compute all distances at once instead of pixel-by-pixel loop
    distances = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(pixels, labels, k):
    new_centroids = []
    for i in range(k):
        clustered_pixels = pixels[labels == i]
        if len(clustered_pixels) > 0:
            new_centroids.append(np.mean(clustered_pixels, axis=0))
        else:
            new_centroids.append(pixels[random.randint(0, len(pixels) - 1)])
    return np.array(new_centroids)


def kmeans(image_path, k=8, iterations=10, resize_factor=1.0, show_result=True,
           savePath=None, medical_warning=True):
    """
    Compress image using K-Means color quantization.

    medical_warning: If True, prints a warning that K-Means is lossy and
                     may not be suitable for diagnostic medical images.
    """
    if medical_warning:
        print("=" * 60)
        print("WARNING: K-Means is a LOSSY compression algorithm.")
        print("Diagnostic details may be lost or altered.")
        print("Not recommended for clinical use — academic analysis only.")
        print("=" * 60)

    if not os.path.exists(image_path):
        print("Image does not exist")
        return None, None

    start_time = time.time()

    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    if resize_factor < 1.0:
        new_size = (int(original_size[0] * resize_factor), int(original_size[1] * resize_factor))
        image = image.resize(new_size)
        print(f"Resized from {original_size} to {new_size}")

    data = np.array(image, dtype=np.float32) / 255.0
    height, width, channels = data.shape
    pixels = data.reshape(-1, 3)

    centroids = initialize_centroids(pixels, k)

    for _ in range(iterations):
        labels = assign_centroids(centroids, pixels)
        new_centroids = update_centroids(pixels, labels, k)
        if np.all(np.abs(centroids - new_centroids) < 0.00001):
            break
        centroids = new_centroids

    cPixels = np.array([centroids[i] for i in labels])
    cImage = cPixels.reshape(height, width, 3)
    cImage_uint8 = (cImage * 255).astype(np.uint8)
    imagePIL = Image.fromarray(cImage_uint8)

    if savePath:
        imagePIL.save(savePath)
        print(f"Image saved to {savePath}")

    runtime = time.time() - start_time

    originalBits = height * width * 24
    compressedBits = k * 24 + height * width * np.log2(k)
    ratio = originalBits / compressedBits

    print(f"Original Bits:     {originalBits:,}")
    print(f"Compressed Bits:   {int(compressedBits):,}")
    print(f"Compression Ratio: {ratio:.4f}")
    print(f"Runtime:           {runtime:.2f}s")

    if show_result:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(imagePIL)
        plt.title(f"K-Means ({k} colors) — LOSSY")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return imagePIL, ratio


def calculate_psnr(originalPath, compressed):
    original = np.array(Image.open(originalPath), dtype=np.float32)
    compressed = np.array(compressed, dtype=np.float32)

    if original.shape != compressed.shape:
        compressed_uint8 = compressed.astype(np.uint8)
        compressed = np.array(
            Image.fromarray(compressed_uint8).resize((original.shape[1], original.shape[0])),
            dtype=np.float32
        )

    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')

    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr


def calculate_ssim(originalPath, compressed):
    """Calculate SSIM between original and K-Means compressed image."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("scikit-image not installed. Run: pip install scikit-image")
        return None

    original = np.array(Image.open(originalPath).convert('RGB'), dtype=np.float32)
    compressed_arr = np.array(compressed, dtype=np.float32)

    if original.shape != compressed_arr.shape:
        compressed_arr = np.array(
            Image.fromarray(compressed_arr.astype(np.uint8)).resize(
                (original.shape[1], original.shape[0])
            ),
            dtype=np.float32
        )

    score = ssim(original, compressed_arr, data_range=255.0, channel_axis=2)
    return score
