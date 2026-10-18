from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import heapq
from collections import Counter


class Node:
    def __init__(self, value=None, frequency=None):
        self.value = value
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency


def huffman_tree(frequency_dictionary):
    heap = [Node(value, frequency) for value, frequency in frequency_dictionary.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        x1 = heapq.heappop(heap)
        x2 = heapq.heappop(heap)
        merged = Node(frequency=x1.frequency + x2.frequency)
        merged.left = x1
        merged.right = x2
        heapq.heappush(heap, merged)

    return heap[0]


def generate_codes(node, curr_code="", codes=None):
    if codes is None:
        codes = {}
    if node.value is not None:
        codes[node.value] = curr_code
        return codes
    if node.left:
        generate_codes(node.left, curr_code + "0", codes)
    if node.right:
        generate_codes(node.right, curr_code + "1", codes)
    return codes


def huffman_encoding(imagePath, savePath=None, bit_depth=8):
    """
    Encode a medical image using Huffman coding.

    bit_depth: 8  = standard grayscale (0-255)
               16 = high-bit medical grayscale (0-65535), common in CT/MRI
    """
    if not os.path.exists(imagePath):
        raise FileNotFoundError(f"Image not found: {imagePath}")

    image = Image.open(imagePath)

    if bit_depth == 16:
        # Convert to 32-bit int mode (PIL's closest to 16-bit), then cast to uint16
        image = image.convert('I')
        pixels = np.array(image, dtype=np.uint16).flatten()
        bits_per_pixel = 16
        print(f"[16-bit mode] Unique intensity levels: {len(np.unique(pixels))}")
    else:
        image = image.convert('L')
        pixels = np.array(image, dtype=np.uint8).flatten()
        bits_per_pixel = 8

    frequency = Counter(int(p) for p in pixels)

    root = huffman_tree(frequency)
    codes = generate_codes(root)

    encoded_data = "".join(codes[int(p)] for p in pixels)

    originalBits = len(pixels) * bits_per_pixel
    compressedBits = len(encoded_data)
    ratio = originalBits / compressedBits

    print(f"Original bits:    {originalBits:,}")
    print(f"Compressed bits:  {compressedBits:,}")
    print(f"Compression ratio: {ratio:.4f}")

    return encoded_data, codes, ratio


def huffman_decoding(encoded_data, codes, image_shape):
    reverseCode = {v: k for k, v in codes.items()}

    decoded_pixels = []
    current_code = ""

    for bit in encoded_data:
        current_code += bit
        if current_code in reverseCode:
            decoded_pixels.append(reverseCode[current_code])
            current_code = ""

    decoded_array = np.array(decoded_pixels, dtype=np.uint8).reshape(image_shape)
    return decoded_array


def calculate_ssim_huffman(imagePath, decoded_array):
    """Calculate SSIM between original and Huffman-decoded image."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("scikit-image not installed. Run: pip install scikit-image")
        return None

    original = np.array(Image.open(imagePath).convert('L'), dtype=np.float32)
    compressed = decoded_array.astype(np.float32)

    if original.shape != compressed.shape:
        compressed = np.array(
            Image.fromarray(compressed).resize((original.shape[1], original.shape[0])),
            dtype=np.float32
        )

    score = ssim(original, compressed, data_range=255.0)
    return score
