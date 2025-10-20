import cv2 as cv
import matplotlib.pyplot as plt


def downsample_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized


def naive_downsample_image(img, factor):
    return img[::factor, ::factor]


def convolve_gaussian(img, kernel_size=5, sigma=1.0):
    return cv.GaussianBlur(img, (kernel_size, kernel_size), sigma)


scale_percent = 10
img = cv.imread("einstein.jpg", cv.IMREAD_GRAYSCALE)
downsampled_img = downsample_image(img, scale_percent)
blurred_img = convolve_gaussian(img, kernel_size=5, sigma=1.0)
downsampled_img = downsample_image(blurred_img, scale_percent)
naive_downsampled_img = naive_downsample_image(img, scale_percent)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.subplot(1, 2, 2)
plt.title(f"Downsampled Image ({scale_percent}%)")
plt.imshow(downsampled_img, cmap="gray")
plt.subplot(1, 2, 1)
plt.title(f"Blurred and Downsampled Image ({scale_percent}%)")
plt.imshow(downsampled_img, cmap="gray")
plt.subplot(1, 2, 2)
plt.title(f"Naively Downsampled Image ({scale_percent}%)")
plt.imshow(naive_downsampled_img, cmap="gray")
plt.show()
