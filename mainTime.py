import cv2
import numpy as np
from Convolution import SignalConvolution

# Apply Normalization to preserve the overall brightness of the image after applying the Gaussian blur.
APPLY_GAUSSIAN_KERNEL_NORMALIZE = True
APPLY_BLUR = True

# Gaussian Blur variables
GAUSSIAN_KERNEL_SIZE = 11  # Must be positive and odd
GAUSSIAN_SIGMA = 1.0

# DOWNSAMPLE variables
DOWNSAMPLE_FACTOR = 8
INPUT_FILE = 'photo.jpg'
OUTPUT_FILE = (f'{DOWNSAMPLE_FACTOR}x-k{GAUSSIAN_KERNEL_SIZE}{"" if APPLY_GAUSSIAN_KERNEL_NORMALIZE else "$"}-'
               f'{"Blur" if APPLY_BLUR else "No Blur"}.jpg')


def create_gaussian_kernel(kernel_size, sigma):
    # middle of the kernel
    k = kernel_size // 2

    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp((-1 * ((x - k) ** 2 + (y - k) ** 2)) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )

    if APPLY_GAUSSIAN_KERNEL_NORMALIZE:
        return kernel / np.sum(kernel)

    else:
        return kernel


def apply_gaussian_blur(image, kernel_size, sigma):
    """Apply Gaussian blur to an image."""

    # Step 1: Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma)

    # Step 2: Apply convolution using convolve2d
    blurred_image = np.zeros_like(image)
    for channel in range(image.shape[2]):
        blurred_image[:, :, channel] = SignalConvolution.convolve2d(image[:, :, channel], kernel, mode='same')

    return blurred_image


# Manually downsample the image
def downsample(image, factor):
    # Calculate the new size
    new_height = image.shape[0] // factor
    new_width = image.shape[1] // factor

    # Initialize the downsampled image
    downsampled_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)

    for y in range(new_height):
        for x in range(new_width):
            for c in range(image.shape[2]):  # Loop over color channels
                downsampled_image[y, x, c] = image[y * factor, x * factor, c]

    return downsampled_image


# Load the image
image = cv2.imread(INPUT_FILE)

# Apply manual Gaussian blur
blurred_image = apply_gaussian_blur(image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)

# Downsample the image manually
downsampled_image = downsample(blurred_image if APPLY_BLUR else image, DOWNSAMPLE_FACTOR)

# Save the downsampled image
cv2.imwrite(OUTPUT_FILE, downsampled_image)

# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(5, 5))
# sns.heatmap(create_gaussian_kernel(KERNEL_SIZE, sigma), annot=True, fmt=".2f", cmap='viridis')
# plt.title('Gaussian Kernel')
# plt.show()
