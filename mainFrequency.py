import cv2
import numpy as np
import fft_dft

# Apply Normalization to preserve the overall brightness of the image after applying the Gaussian blur.
APPLY_GAUSSIAN_KERNEL_NORMALIZE = True
APPLY_BLUR = True

# Gaussian Blur variables
GAUSSIAN_KERNEL_SIZE = 9  # Must be positive and odd
GAUSSIAN_SIGMA = 1.0

# DOWNSAMPLE variables
DOWNSAMPLE_FACTOR = 9
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

def apply_fft_gaussian_blur(image, kernel_size, sigma):
    from scipy.fft import \
    (fft2, ifft2, fftshift, ifftshift)

    # Step 1: Create Gaussian kernel in spatial domain
    kernel = create_gaussian_kernel(kernel_size, sigma)

    # Expand kernel to match image dimensions
    if len(image.shape) == 3:
        kernel = np.stack([kernel] * image.shape[2], axis=2)

    # Step 2: Compute FFT of the image
    image_fft = fft2(image, axes=(0, 1))
    image_fft_shifted = fftshift(image_fft, axes=(0, 1))

    # Step 3: Compute FFT of the Gaussian kernel
    kernel_fft = fft2(kernel, s=image.shape[:2], axes=(0, 1))
    kernel_fft_shifted = fftshift(kernel_fft, axes=(0, 1))

    # Step 4: Apply Gaussian blur in frequency domain
    blurred_fft_shifted = image_fft_shifted * kernel_fft_shifted
    blurred_fft = ifftshift(blurred_fft_shifted, axes=(0, 1))
    blurred_image = np.real(ifft2(blurred_fft, axes=(0, 1)))

    return blurred_image.astype(np.uint8)


def downsample(image, factor):
    # Calculate the new size
    new_height = image.shape[0] // factor
    new_width = image.shape[1] // factor
    # Downsample the image
    downsampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return downsampled_image


# Load the image
image = cv2.imread(INPUT_FILE)

# Apply FFT-based Gaussian blur
blurred_image = apply_fft_gaussian_blur(image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)

# Downsample the blurred image
downsampled_image = downsample(blurred_image if APPLY_BLUR else image, DOWNSAMPLE_FACTOR)

# Save the downsampled image
cv2.imwrite(OUTPUT_FILE, downsampled_image)
