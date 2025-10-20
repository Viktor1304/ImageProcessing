import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread("ely.jpeg", cv.IMREAD_GRAYSCALE)
print(np.uint8(img))

# 2. Apply Sobel filters (Gx and Gy)
sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)  # derivative in x-direction
sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)  # derivative in y-direction

# 3. Compute the gradient magnitude (the "general form" |Δ(G * I)|)
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Normalize for display (0–255)
gradient_magnitude = cv.normalize(
    gradient_magnitude, gradient_magnitude, 0, 255, cv.NORM_MINMAX
)

gradient_magnitude = np.uint8(gradient_magnitude)

gradient_magnitude_x = np.sqrt(sobel_x**2)
gradient_magnitude_x = cv.normalize(
    gradient_magnitude_x, gradient_magnitude_x, 0, 255, cv.NORM_MINMAX
)

gradient_magnitude_y = np.sqrt(sobel_y**2)
gradient_magnitude_y = cv.normalize(
    gradient_magnitude_y, gradient_magnitude_y, 0, 255, cv.NORM_MINMAX
)
# 4. Show results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray"), plt.title("Original Image")
(
    plt.subplot(1, 3, 2),
    plt.imshow(gradient_magnitude_x, cmap="gray"),
    plt.title("Sobel X (∂I/∂x)"),
)
(
    plt.subplot(1, 3, 3),
    plt.imshow(gradient_magnitude_y, cmap="gray"),
    plt.title("Sobel Y (∂I/∂y)"),
)
plt.figure()
plt.imshow(gradient_magnitude, cmap="gray"), plt.title("Gradient Magnitude |Δ(G*I)|")
plt.show()
