from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time


def RLE_encoding(imagePath):
    if not os.path.exists(imagePath):
        print("File does not exist")
        return None

    image = Image.open(imagePath).convert("RGB")
    imageArray = np.array(image, dtype=np.uint8)
    height, width, channel = imageArray.shape

    pixelsFlat = [tuple(pixel) for pixel in imageArray.reshape(-1, 3)]

    start = time.time()

    encoded_list = []
    prevPixel = pixelsFlat[0]
    count = 1

    for pixel in pixelsFlat[1:]:
        if pixel == prevPixel:
            count += 1
        else:
            encoded_list.append((prevPixel, count))
            prevPixel = pixel
            count = 1
    encoded_list.append((prevPixel, count))

    runtime = time.time() - start

    originalBits = len(pixelsFlat) * 24
    compressedBits = len(encoded_list) * (24 + 16)
    ratio = originalBits / compressedBits

    print(f"Original bits:     {originalBits:,}")
    print(f"Compressed bits:   {compressedBits:,}")
    print(f"Compression ratio: {ratio:.4f}")
    print(f"Runtime:           {runtime:.4f}s")

    return encoded_list, (height, width, channel), ratio, runtime


def RLE_decoding(encoded_list, imageShape):
    height, width, channel = imageShape
    pixels = []
    for pixel, count in encoded_list:
        pixels.extend([pixel] * count)

    pixels = np.array(pixels[:height * width], dtype=np.uint8).reshape((height, width, channel))
    return pixels


def calculate_ssim_rle(originalPath, decoded_array):
    """Calculate SSIM between original and RLE-decoded image."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("scikit-image not installed. Run: pip install scikit-image")
        return None

    original = np.array(Image.open(originalPath).convert('RGB'), dtype=np.float32)
    compressed = decoded_array.astype(np.float32)

    score = ssim(original, compressed, data_range=255.0, channel_axis=2)
    return score


def calculate_psnr_rle(originalPath, decoded_array):
    original = np.array(Image.open(originalPath).convert('RGB'), dtype=np.float32)
    compressed = decoded_array.astype(np.float32)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def run_RLE(imagePath, savePath=None):
    result = RLE_encoding(imagePath)
    if result is None:
        return None, None, None

    encoded_list, image_shape, ratio, runtime = result
    decoded_array = RLE_decoding(encoded_list, image_shape)
    decoded_image = Image.fromarray(decoded_array)

    if savePath:
        decoded_image.save(savePath)
        print(f"Image saved to {savePath}")

    original_image = Image.open(imagePath).convert("RGB")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(decoded_image)
    plt.title("RLE Compressed Image (Lossless)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return decoded_image, ratio, runtime
