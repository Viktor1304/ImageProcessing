import random
import matplotlib.pyplot as plt


NMB_PIXELS = 1000
pixels: list[int] = [int(random.random() * 128) for _ in range(int(NMB_PIXELS / 5))]
pixels.extend(
    [int(random.random() * 128) + 128 for _ in range(int(4 * NMB_PIXELS / 5))]
)

max_int = max(pixels)
cumulative: dict[int, int] = {}

for pixel in pixels:
    for idx in range(pixel):
        if idx not in cumulative.keys():
            cumulative[idx] = 0
        cumulative[idx] += 1
x_values: list[int] = list(cumulative.keys())
y_values: list[int] = list(cumulative.values())

plt.plot(x_values, y_values)
plt.show()

new_pixels: list[int] = [
    int(max_int * cumulative[max(0, pixel - 1)] / NMB_PIXELS) for pixel in pixels
]

new_cumulative: dict[int, int] = {}
for pixel in new_pixels:
    for idx in range(pixel):
        if idx not in new_cumulative.keys():
            new_cumulative[idx] = 0
        new_cumulative[idx] += 1

print(new_cumulative)
x_values = list(new_cumulative.keys())
y_values = list(new_cumulative.values())

plt.plot(x_values, y_values)
plt.show()
