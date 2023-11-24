import cv2
import re
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import time
from scipy.signal import convolve2d

def load_images():
    """
    This method iterates over all the files in the output directory (generated from binarization) and picks the images from that directory
    """
    current_directory = os.getcwd()
    dataset_directory = "output"
    image_directory = os.path.join(current_directory, dataset_directory)
    all_files = os.listdir(image_directory)
    image_files = [f for f in all_files if f.lower().endswith(('.png'))]
    image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0])) # sort the images by digit
    print(f"Total images found in the dataset: {len(image_files)}")
    return image_files, image_directory

def image_to_array(images: list, image_directory: str):
    """
    This method iterates over each image and converts the image into RGBA and saves them in an np array
    """
    image_arrays = []
    for image in images:
        path = image_directory + "/" + image
        image_read = cv2.imread(path) # read an image
        image_rgb = cv2.cvtColor(image_read, cv2.COLOR_BGR2RGB) # convert the image into RGBA  
        image_arrays.append(image_rgb) # add each RGBA image to the list

    image_np_arrays = np.array(image_arrays)
    return image_np_arrays

def gaussian_kernel(size, sigma=1.0):
    """
    This method generates a 2D Gaussian Kernel needed for gaussian blurring functions
    """

    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )

    return kernel / np.sum(kernel)

def sequential_gaussian_blur_image(image, kernel_size=5, sigma=1.0):
    """
    This function applies the gaussian blur to image using a sequential method
    """

    start_time = time.time()

    kernel = gaussian_kernel(kernel_size, sigma)

    blurred_image = np.zeros_like(image, dtype=float)
    for c in range(image.shape[-1]):
        blurred_image[..., c] = convolve2d(image[..., c], kernel, mode='same', boundary='symm')

    return blurred_image.astype(np.uint8), time.time() - start_time

def sequential_gaussian_blur_all(image_np_arrays):

    """
    Iterates through all the image np arrays and applies the gaussian blur technique
    """

    current_directory = os.getcwd()
    output_directory = "final_output"
    output_directory = os.path.join(current_directory, output_directory)
    os.makedirs(output_directory, exist_ok=True)

    execution_time_array = []

    counter = 1

    for image in image_np_arrays:

        gaussian_blurred_image, execution_time = sequential_gaussian_blur_image(image)

        output_path = os.path.join(output_directory, f"binarized&gaussianBlur_{counter}.png")
        cv2.imwrite(output_path, gaussian_blurred_image) # store the binarized image into the output directory

        execution_time_array.append(execution_time)

        counter = counter + 1

    return execution_time_array

def plot_execution_time(execution_time_array):
    """
    This method plots the execution time of the binarization on the number of images
    """
    plt.title("Execution time for each image")
    plt.scatter(np.arange(start=1, stop=101, step=1), np.array(execution_time_array), color="blue")
    plt.axhline(y=np.nanmean(np.array(execution_time_array)), color="red")
    legend_elements = [Line2D([0], [0], color='r', lw=4, label='Mean'), Line2D([0], [0], marker='o', color="w", label='Individual Execution Time', markerfacecolor='b', markersize=15),]
    plt.legend(handles=legend_elements, loc="upper left")
    plt.xlabel("Image Number")
    plt.ylabel("Time (s)")

    plt.savefig("analysis/gaussianBlur_sequential_execution_time.png")
    plt.close("analysis/gaussianBlur_sequential_execution_time.png")

    print("--- Sequential Gaussian Blur Execution Time: %s seconds ---" % (sum(execution_time_array)))


def main():
    # images contains filenames of all binarized png images
    # image_directory contains the complete directory address for the output folder
    images, image_directory = load_images()
    image_np_arrays = image_to_array(images, image_directory)
    execution_time_array = sequential_gaussian_blur_all(image_np_arrays)
    plot_execution_time(execution_time_array)

if __name__ == "__main__":
    main()