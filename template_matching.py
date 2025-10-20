import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt

NDArrayInt8 = npt.NDArray[np.uint8]
NDArrayFloat64 = npt.NDArray[np.float64]


def normalized_cross_correlation(image: NDArrayInt8, template: NDArrayInt8):
    h, w = template.shape

    template_mean = np.mean(template)
    template_zero_mean = template - template_mean

    output = np.zeros((image.shape[0] - h + 1, image.shape[1] - w + 1))

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            patch = image[i : i + h, j : j + w]
            patch_mean = np.mean(patch)
            patch_zero_mean = patch - patch_mean

            numerator = np.sum(template_zero_mean * patch_zero_mean)
            denominator: np.float128 = np.float128(
                np.sqrt(np.sum(template_zero_mean**2) * np.sum(patch_zero_mean**2))
            )

            output[i, j] = 0
            if denominator != 0:
                output[i, j] = numerator / denominator

    return output


def mean_cross_correlation(image: NDArrayInt8, template: NDArrayInt8):
    h, w = template.shape

    template_mean = np.zeros_like(template)
    template_zero_mean = template - template_mean

    output = np.zeros((image.shape[0] - h + 1, image.shape[1] - w + 1))

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            patch = image[i : i + h, j : j + w]
            patch_mean = np.mean(patch)
            patch_zero_mean = patch - patch_mean

            numerator = np.sum(template_zero_mean * patch_zero_mean)
            denominator: NDArrayFloat64 = np.sqrt(
                np.sum(template_zero_mean**2) * np.sum(patch_zero_mean**2)
            )

            output[i, j] = 0
            if denominator != 0:
                output[i, j] = numerator / denominator

    return output


def cross_correlation(image, template):
    h, w = template.shape

    template_mean = np.zeros_like(template)
    template_zero_mean = template - template_mean

    output = np.zeros((image.shape[0] - h + 1, image.shape[1] - w + 1))

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            patch = image[i : i + h, j : j + w]
            patch_mean = np.zeros_like(patch)
            patch_zero_mean = patch - patch_mean

            numerator = np.sum(template_zero_mean * patch_zero_mean)
            denominator = np.sqrt(
                np.sum(template_zero_mean**2) * np.sum(patch_zero_mean**2)
            )

            output[i, j] = 0
            if denominator != 0:
                output[i, j] = numerator / denominator

    return output


def threshold(img, threshold: float = 200):
    output = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if abs(img[i, j] * 255) < threshold:
                output[i, j] = 0

    return output


if __name__ == "__main__":
    img = cv.imread("ely.jpeg", cv.IMREAD_GRAYSCALE)

    print(f"Image shape is {img.shape}")  # pyright: ignore[reportOptionalMemberAccess, reportAny]

    template = img[0:50, 0:100]

    output = normalized_cross_correlation(img, template)
    print(output)
    output = threshold(output)
    print("Normalized cross correlation computed")
    no_mean_output = mean_cross_correlation(img, template)
    no_mean_output = threshold(no_mean_output)
    print("Mean cross correlation computed")
    cross_output = cross_correlation(img, template)
    cross_output = threshold(cross_output)
    print("Cross correlation computed")

    # 4. Show results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray"), plt.title("Original Image")
    (
        plt.subplot(1, 3, 2),
        plt.imshow(template, cmap="gray"),
        plt.title("Template"),
    )
    (
        plt.subplot(6, 3, 3),
        plt.imshow(output, cmap="gray"),
        plt.title("Ouput of normalized cross correlation"),
    )
    (
        plt.subplot(6, 3, 1),
        plt.imshow(no_mean_output, cmap="gray"),
        plt.title("Ouput of mean cross correlation"),
    )
    (
        plt.subplot(6, 3, 2),
        plt.imshow(cross_output, cmap="gray"),
        plt.title("Ouput of mean cross correlation"),
    )
    plt.show()
