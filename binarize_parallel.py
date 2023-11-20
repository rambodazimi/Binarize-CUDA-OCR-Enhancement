import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import random
import time
import cupy as cp


def load_images():
    """
    This method iterates over all the files in the dataset directory and picks the images from that directory
    """
    current_directory = os.getcwd()
    dataset_directory = "dataset"
    image_directory = os.path.join(current_directory, dataset_directory)
    all_files = os.listdir(image_directory)
    image_files = [f for f in all_files if f.lower().endswith(('.png'))]
    image_files.sort(key=lambda x: int(x.split('.')[0])) # sort the images by digit
    print(f"Total images found in the dataset: {len(image_files)}")
    return image_files, image_directory

def image_to_array(images: list, image_directory: str):
    """
    This method iterates over each image and converts the image into RGBA and saves them in a cupy array
    """
    image_arrays = []
    for image in images:
        path = image_directory + "/" + image
        image_read = cv2.imread(path) # read an image
        image_rgb = cv2.cvtColor(image_read, cv2.COLOR_BGR2RGB) # convert the image into RGBA  
        image_arrays.append(image_rgb) # add each RGBA image to the list

    image_np_arrays = np.array(image_arrays)
    image_cupy_arrays = cp.asarray(image_np_arrays)
    return image_cupy_arrays

def binarize_image_cupy(image, threshold):
    """
    This method applies binarization on a given image in cupy format
    """
    start_time = time.time()
    # Convert pixel to grayscale intensity using the formula: intensity = 0.299*R + 0.587*G + 0.114*B
    grayscale_intensity = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    # Binarize based on the grayscale intensity
    binarized_result = cp.where(grayscale_intensity > threshold, 255, 0)
    return binarized_result, time.time() - start_time

def binarize_all_parallel(image_cuda_arrays):
    """
    This method applies binarization to all the images in the dataset
    """
    current_directory = os.getcwd()
    output_directory = "output"
    output_directory = os.path.join(current_directory, output_directory)

    execution_time_arrays = []
    counter = 1
    for image in image_cuda_arrays:
        binarized_image = binarize_image_cupy(image, 127)
        result = binarized_image[0] # threshold is set to 127
        execution_time_arrays.append(binarized_image[1])
        output_path = os.path.join(output_directory, f"binarized_{counter}.png")
        cv2.imwrite(output_path, cp.asnumpy(result)) # store the binarized image into the output directory
        counter = counter + 1

    print("--- Parallel Binarization Execution Time: %s seconds ---" % (sum(execution_time_arrays)))

    return execution_time_arrays


def plot_execution_time(image_cupy_arrays):
    """
    This method plots the execution time of the binarization on the number of images
    """
    result = binarize_all_parallel(image_cupy_arrays)
    plt.title("Execution time for each image")
    plt.scatter(np.arange(start=1, stop=101, step=1), np.array(result), color="blue")
    plt.axhline(y=np.nanmean(np.array(result)), color="red")
    legend_elements = [Line2D([0], [0], color='r', lw=4, label='Mean'), Line2D([0], [0], marker='o', color="w", label='Individual Execution Time', markerfacecolor='b', markersize=15),]
    plt.legend(handles=legend_elements, loc="upper left")
    plt.xlabel("Image Number")
    plt.ylabel("Time (s)")

    plt.savefig("analysis/parallel_execution_time.png")
    plt.close("analysis/parallel_execution_time.png")


def main():
    images, image_directory = load_images()
    image_cupy_arrays = image_to_array(images, image_directory) # each index of this array contains an image in cupy_array format
    plot_execution_time(image_cupy_arrays)


if __name__ == "__main__":
    main()