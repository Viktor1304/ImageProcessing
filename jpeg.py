from typing import TypeAlias
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt


NDArrayUint16: TypeAlias = npt.NDArray[np.uint16]
NDArrayInt16: TypeAlias = npt.NDArray[np.int16]
NDArrayFloat64: TypeAlias = npt.NDArray[np.float64]


class QuantizationMatrix:
    def __init__(self, quality: int):
        self.matrix: NDArrayUint16
        if quality == 90:
            self.matrix = np.array(
                [
                    [3, 2, 2, 3, 5, 8, 10, 12],
                    [2, 2, 3, 4, 5, 12, 12, 11],
                    [3, 3, 3, 5, 8, 11, 14, 11],
                    [3, 3, 4, 6, 10, 17, 16, 12],
                    [4, 4, 7, 11, 14, 22, 21, 15],
                    [5, 7, 11, 13, 16, 12, 23, 18],
                    [10, 13, 16, 17, 21, 24, 24, 21],
                    [14, 18, 19, 20, 22, 20, 20, 20],
                ]
            )
        elif quality == 10:
            self.matrix = np.array(
                [
                    [80, 60, 50, 80, 120, 200, 255, 255],
                    [60, 60, 70, 95, 130, 255, 255, 255],
                    [70, 65, 80, 120, 200, 255, 255, 255],
                    [70, 85, 110, 145, 255, 255, 255, 255],
                    [90, 110, 185, 255, 255, 255, 255, 255],
                    [120, 175, 255, 255, 255, 255, 255, 255],
                    [245, 255, 255, 255, 255, 255, 255, 255],
                    [255, 255, 255, 255, 255, 255, 255, 255],
                ]
            )
        else:
            self.matrix = np.array(
                [
                    [16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99],
                ]
            )


def read_jpeg_image(file_path: str) -> NDArrayUint16:
    """
    Reads a JPEG image from the specified file path.

    Args:
        file_path (str): The path to the JPEG image file.
    Returns:
        np.ndarray: The image as a NumPy array.
    """

    image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {file_path}")
    return np.array(image, dtype=np.uint16)


def divide_image_into_blocks(
    image: NDArrayUint16, block_size: int
) -> list[NDArrayUint16]:
    """
    Divides the image into non-overlapping blocks of specified size.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        block_size (int): The size of each block (block_size x block_size).
    Returns:
        list: A list of image blocks as NumPy arrays.
    """

    h, w = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block: NDArrayUint16 = image[i : i + block_size, j : j + block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                blocks.append(block)
            else:
                padded_block: NDArrayUint16 = np.zeros(
                    (block_size, block_size), dtype=image.dtype
                )
                padded_block[: block.shape[0], : block.shape[1]] = block
                blocks.append(padded_block)
    return blocks


def level_shift_block(block: NDArrayUint16) -> NDArrayInt16:
    """
    Applies level shifting to the image block by subtracting 128 from each pixel value.

    Args:
        block (np.ndarray): The input image block as a NumPy array.
    Returns:
        np.ndarray: The level-shifted image block.
    """

    return block.astype(np.int16) - 128


def DCT_matrix(N: int) -> NDArrayFloat64:
    """
    Generates the DCT transformation matrix of size N x N.

    Args:
        N (int): The size of the DCT matrix.
    Returns:
        np.ndarray: The DCT transformation matrix.
    """

    dct_matrix = np.zeros((N, N))
    for u in range(N):
        for x in range(N):
            cu: np.float64 = np.sqrt(2 / N)
            if u == 0:
                cu = 1 / np.sqrt(N)
            dct_matrix[u, x] = cu * np.cos(((2 * x + 1) * u * np.pi) / (2 * N))
    return dct_matrix


def apply_DCT(block: NDArrayInt16) -> NDArrayFloat64:
    """
    Applies the Discrete Cosine Transform (DCT) to the image block.

    Args:
        block (np.ndarray): The input image block as a NumPy array.
    Returns:
        np.ndarray: The DCT-transformed image block.
    """

    N = block.shape[0]
    dct_matrix = DCT_matrix(N)
    dct_block = np.dot(dct_matrix, np.dot(block, dct_matrix.T))
    return dct_block


def quantize_block(
    dct_block: NDArrayFloat64, quant_matrix: NDArrayUint16
) -> NDArrayInt16:
    """
    Quantizes the DCT-transformed image block using the provided quantization matrix.

    Args:
        dct_block (np.ndarray): The DCT-transformed image block.
        quant_matrix (np.ndarray): The quantization matrix.
    Returns:
        np.ndarray: The quantized image block.
    """

    quantized_block = np.round(dct_block / quant_matrix).astype(np.int16)
    return quantized_block


def encode(quantized_block: NDArrayInt16) -> list[tuple[int, int]]:
    """
    Encodes the quantized image block using a simple run-length encoding (RLE) scheme.

    Args:
        quantized_block (np.ndarray): The quantized image block.
    Returns:
        list: The RLE-encoded data as a list of tuples (value, count).
    """

    flat_block = quantized_block.flatten()
    encoded_data = []
    count = 1
    for i in range(1, len(flat_block)):
        if flat_block[i] == flat_block[i - 1]:
            count += 1
        else:
            encoded_data.append((flat_block[i - 1], count))
            count = 1
    encoded_data.append((flat_block[-1], count))

    return encoded_data


def decompression(
    encoded_data: list[tuple[int, int]], block_size: int, quant_matrix: NDArrayUint16
) -> NDArrayUint16:
    """
    Decompresses the encoded data back into the image block.

    Args:
        encoded_data (list): The RLE-encoded data as a list of tuples (value, count).
        block_size (int): The size of the image block.
        quant_matrix (np.ndarray): The quantization matrix.
    Returns:
        np.ndarray: The decompressed image block.
    """

    flat_block = []
    for value, count in encoded_data:
        flat_block.extend([value] * count)
    quantized_block = np.array(flat_block).reshape((block_size, block_size))
    dequantized_block = quantized_block * quant_matrix
    return dequantized_block


def apply_IDCT(dct_block: NDArrayUint16) -> NDArrayFloat64:
    """
    Applies the Inverse Discrete Cosine Transform (IDCT) to the DCT-transformed image block.
    Args:
        dct_block (np.ndarray): The DCT-transformed image block.
    Returns:
        np.ndarray: The IDCT-transformed image block.
    """

    N = dct_block.shape[0]
    dct_matrix = DCT_matrix(N)
    idct_block = np.dot(dct_matrix.T, np.dot(dct_block, dct_matrix))
    return idct_block


def pad_iamage(image: NDArrayUint16, block_size: int) -> tuple[NDArrayUint16, int, int]:
    """
    Pads the image to make its dimensions multiples of block_size.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        block_size (int): The block size for padding.
    Returns:
        np.ndarray: The padded image.
    """

    h, w = image.shape
    if h % block_size != 0:
        pad_height = block_size - (h % block_size)
        image = np.pad(
            image, ((0, pad_height), (0, 0)), mode="constant", constant_values=0
        )
        h += pad_height
    if w % block_size != 0:
        pad_width = block_size - (w % block_size)
        image = np.pad(
            image, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
        )
        w += pad_width

    return (image, h, w)


def apply_jpeg(img: NDArrayUint16, quality: int = 50):
    """
    Applies JPEG compression and decompression to the input image.

    Args:
        img (np.ndarray): The input image as a NumPy array.
    Returns:
        np.ndarray: The reconstructed image after JPEG compression and decompression.
    """

    block_size = 8
    quant_matrix = QuantizationMatrix(quality=quality)

    img, h, w = pad_iamage(img, block_size)
    reconstructed_img = np.zeros((h, w), dtype=np.uint8)

    blocks = divide_image_into_blocks(img, block_size)

    def estimate_compression_ratio(
        blocks: list[NDArrayUint16], h: int, w: int, quant_matrix: QuantizationMatrix
    ) -> float:
        original_size = h * w
        total_encoded_size = 0
        for block in blocks:
            level_shifted = level_shift_block(block)
            dct_block = apply_DCT(level_shifted)
            quantized_block = quantize_block(
                dct_block, quant_matrix=quant_matrix.matrix
            )
            encoded_data = encode(quantized_block)
            total_encoded_size += len(encoded_data) * 2  # Each tuple (value, count)
        compression_ratio = original_size / total_encoded_size
        return compression_ratio

    print(
        f"Compression Ratio: {estimate_compression_ratio(blocks, h, w, quant_matrix):.2f}"
    )

    for idx, block in enumerate(blocks):
        level_shifted = level_shift_block(block)
        dct_block = apply_DCT(level_shifted)
        quantized_block = quantize_block(dct_block, quant_matrix=quant_matrix.matrix)
        encoded_data = encode(quantized_block)
        decompressed_dct_block = decompression(
            encoded_data, block_size, quant_matrix=quant_matrix.matrix
        )
        idct_block = apply_IDCT(decompressed_dct_block)
        idct_block += 128
        idct_block = np.clip(idct_block, 0, 255).astype(np.uint8)

        i = (idx * block_size) // w * block_size
        j = (idx * block_size) % w
        reconstructed_img[i : i + block_size, j : j + block_size] = idct_block

    return reconstructed_img


if __name__ == "__main__":
    input_image_path = "your_image.jpg"
    image = read_jpeg_image(input_image_path)
    reconstructed_image = apply_jpeg(image)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_image, cmap="gray")
    plt.axis("off")
    plt.show()
