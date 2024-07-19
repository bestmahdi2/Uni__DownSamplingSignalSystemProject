import numpy as np
from typing import List
import matplotlib.pyplot as plt

# File name to save the plot
FILE_NAME: str = 'Results.png'


class SignalConvolution:
    """
    Class to perform signal convolution, both discrete and continuous.
    """

    def __init__(self, accuracy: int) -> None:
        """
        Initialize SignalConvolution class.

        Parameters:
        - accuracy (int): Number of samples per second.
        """
        self.accuracy: int = accuracy

    @staticmethod
    def convolve2d( image: np.ndarray, kernel: np.ndarray, mode: str = 'same') -> np.ndarray:
        """
        Perform 2D convolution on the image with the given kernel.

        Parameters:
        - image (np.ndarray): Input image (2D NumPy array).
        - kernel (np.ndarray): Convolution kernel (2D NumPy array).
        - mode (str): Padding mode ('same', 'valid', or 'full'). Default is 'same'.

        Returns:
        - np.ndarray: Convolved image (2D NumPy array).
        """
        # Get dimensions of image and kernel
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # Flip the kernel (180-degree rotation)
        kernel = np.flipud(np.fliplr(kernel))

        # Determine padding based on selected mode
        if mode == 'same':
            pad_height = kernel_height // 2
            pad_width = kernel_width // 2
        elif mode == 'valid':
            pad_height = 0
            pad_width = 0
        elif mode == 'full':
            pad_height = kernel_height - 1
            pad_width = kernel_width - 1
        else:
            raise ValueError("Unsupported padding mode. Choose 'same', 'valid', or 'full'.")

        # Pad the image
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        # Initialize the result array
        result_height = image_height + 2 * pad_height - kernel_height + 1
        result_width = image_width + 2 * pad_width - kernel_width + 1
        result = np.zeros((result_height, result_width))

        # Perform 2D convolution
        for i in range(result_height):
            for j in range(result_width):
                # Extract region of interest (ROI) from padded image
                roi = padded_image[i:i + kernel_height, j:j + kernel_width]
                # Perform element-wise multiplication and sum
                result[i, j] = np.sum(roi * kernel)

        return result

    def convolution_discrete(self, x_signal: List[float], x_start: float, h_signal: List[float],
                             h_start: float) -> None:
        """
        Perform discrete convolution and plot the signals.

        Parameters:
        - x_signal (List[float]): Input signal x(t).
        - x_start (float): Start time of signal x(t).
        - h_signal (List[float]): Input signal h(t).
        - h_start (float): Start time of signal h(t).

        Returns:
        - None
        """
        # Convert start times to discrete indices
        x_start *= self.accuracy
        h_start *= self.accuracy

        # Calculate the length of the resulting sequence
        result_length: int = len(x_signal) + len(h_signal) - 1

        # Initialize the result sequence with zeros
        result: List[float] = [0] * result_length

        # Perform discrete convolution
        for i in range(len(x_signal)):
            for j in range(len(h_signal)):
                result[i + j] += x_signal[i] * h_signal[j]

        # Adjust the start index of the convolution result
        conv_start: float = x_start + h_start

        # Plot the signals and the convolution result
        plt.figure(figsize=(10, 6))

        # Plot x(t)
        plt.subplot(3, 1, 1)
        plt.stem(range(int(x_start), int(x_start + len(x_signal))), x_signal)
        plt.title('x(t)')
        plt.xlabel('t')
        plt.ylabel('Amplitude')

        # Plot h(t)
        plt.subplot(3, 1, 2)
        plt.stem(range(int(h_start), int(h_start + len(h_signal))), h_signal)
        plt.title('h(t)')
        plt.xlabel('t')
        plt.ylabel('Amplitude')

        # Plot the convolution result
        plt.subplot(3, 1, 3)
        plt.stem(range(int(conv_start), int(conv_start + result_length)), result)
        plt.title('x(t) * h(t)')
        plt.xlabel('t')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.savefig(FILE_NAME)
        plt.show()

        # Print the mathematical result of convolution
        print("Mathematical result of convolution (y[n]):")
        print(result)

    def convolution_continues(self, x_signal: List[float], x_start: float, h_signal: List[float],
                              h_start: float) -> None:
        """
        Perform continuous convolution and plot the signals.

        Parameters:
        - x_signal (List[float]): Input signal x(t).
        - x_start (float): Start time of signal x(t).
        - h_signal (List[float]): Input signal h(t).
        - h_start (float): Start time of signal h(t).

        Returns:
        - None
        """
        x_start *= self.accuracy
        h_start *= self.accuracy

        result: List[List[float]] = []
        for num in h_signal:
            result.append([num * i for i in x_signal])

        for i in range(len(h_signal)):
            result[i] = [0] * i + result[i] + [0] * (len(h_signal) - 1 - i)

        result = list(zip(*result))
        conv_result: List[float] = [sum(i) for i in result]

        conv_start: float = x_start + h_start

        x_signal = [0.0] * self.accuracy + x_signal + [0.0] * self.accuracy
        h_signal = [0.0] * self.accuracy + h_signal + [0.0] * self.accuracy
        conv_result = [0.0] * self.accuracy + conv_result + [0.0] * self.accuracy

        fig, diagram = plt.subplots(3, 1, sharex=True)

        diagram[0].plot(
            [i / self.accuracy for i in
             range(int(x_start - self.accuracy), int(x_start + len(x_signal) - self.accuracy))],
            x_signal,
            label='x(t)')

        diagram[1].plot(
            [i / self.accuracy for i in
             range(int(h_start - self.accuracy), int(h_start + len(h_signal) - self.accuracy))],
            h_signal,
            label='h(t)')

        diagram[2].plot([i / self.accuracy for i in
                         range(int(conv_start - self.accuracy), int(conv_start + len(conv_result) - self.accuracy))],
                        [i / self.accuracy for i in conv_result], label='x(t) * h(t)')

        for axis in diagram:
            axis.legend()

        plt.savefig(FILE_NAME)
        plt.show()

        # Print the mathematical result of convolution
        print("Mathematical result of convolution (y[n]):")
        print(result)


def main() -> None:
    """
    Main function to execute the program.
    """
    # Ask user for choice
    function: str = input("Which one do you prefer? [1 or 2]\n1. Continuous\n2. Discrete\n > ")
    choice: str = input("Use default signals? (Y/N) [N=enter each parameters by yourself]\n > ").strip().lower()

    if choice.lower() == 'y':
        # Number of samples per second
        accuracy: int = 50

        # Signal x(t) parameters
        start_x: float = 0
        end_x: float = 2
        x_signal: List[float] = [2.0] * (int(end_x * accuracy) - int(start_x * accuracy))

        # Signal h(t) parameters
        start_h: float = 0
        end_h: float = 1
        h_signal: List[float] = [3.0] * (int(end_h * accuracy) - int(start_h * accuracy))

    else:
        # Number of samples per second
        accuracy: int = int(input("> Number of samples per second [Enter=50]: ") or 50)

        # Signal x(t) parameters
        start_x: float = int(input("> x(t) start time [Enter=0]: ") or 0)
        end_x: float = int(input("> x(t) end time [Enter=2]: ") or 2)
        x_length: int = int(end_x * accuracy) - int(start_x * accuracy)
        x_signal: List[float] = [float(x) for x in
                                 input(
                                     f"> Enter values for signal x(t) separated by spaces ({x_length} values)[Enter=all 2]: ").split()] or [
                                    2] * x_length

        # Signal h(t) parameters
        start_h: float = int(input("> h(t) start time [Enter=0]: ") or 0)
        end_h: float = int(input("> h(t) end time [Enter=1]: ") or 1)
        h_length: int = int(end_h * accuracy) - int(start_h * accuracy)
        h_signal: List[float] = [float(x) for x in
                                 input(
                                     f"> Enter values for signal h(t) separated by spaces ({h_length} values)[Enter=all 3]: ").split()] or [
                                    3] * h_length

    # Create an instance of SignalConvolution
    signal_conv: SignalConvolution = SignalConvolution(accuracy)

    if function == "1":
        # Perform continuous convolution
        signal_conv.convolution_continues(x_signal, start_x, h_signal, start_h)
    else:
        # Perform discrete convolution
        signal_conv.convolution_discrete(x_signal, start_x, h_signal, start_h)


if __name__ == "__main__":
    main()


def convolve2d(image, kernel, mode='same'):
    """
    Perform 2D convolution on the image with the given kernel.

    Parameters:
    - image: Input image (2D NumPy array).
    - kernel: Convolution kernel (2D NumPy array).
    - mode: Padding mode ('same', 'valid', or 'full'). Default is 'same'.

    Returns:
    - result: Convolved image (2D NumPy array).
    """
    # Get dimensions of image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Flip the kernel (180 degree rotation)
    kernel = np.flipud(np.fliplr(kernel))

    # Determine padding based on selected mode
    if mode == 'same':
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
    elif mode == 'valid':
        pad_height = 0
        pad_width = 0
    elif mode == 'full':
        pad_height = kernel_height - 1
        pad_width = kernel_width - 1
    else:
        raise ValueError("Unsupported padding mode. Choose 'same', 'valid', or 'full'.")

    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initialize the result array
    result_height = image_height + 2 * pad_height - kernel_height + 1
    result_width = image_width + 2 * pad_width - kernel_width + 1
    result = np.zeros((result_height, result_width))

    # Perform 2D convolution
    for i in range(result_height):
        for j in range(result_width):
            # Extract region of interest (ROI) from padded image
            roi = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Perform element-wise multiplication and sum
            result[i, j] = np.sum(roi * kernel)

    return result
