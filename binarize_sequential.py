import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

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

def binarize_image(image: np.array, threshold: int):
    """
    This method applies binarization with a given threshold to the given image and returns the copy of that image
    """
    image_copy = image.copy()  # Make a copy of the image

    if len(image_copy.shape) == 3:  # Check if the image is RGB
        height, width, _ = image_copy.shape
        for h in range(height):
            for w in range(width):
                # Convert pixel to grayscale intensity using the formula: intensity = 0.299*R + 0.587*G + 0.114*B
                grayscale_intensity = 0.299 * image_copy[h, w, 0] + 0.587 * image_copy[h, w, 1] + 0.114 * image_copy[h, w, 2]
                
                # Binarize based on the grayscale intensity
                image_copy[h, w] = 255 if grayscale_intensity > threshold else 0
    else:
        raise ValueError("Invalid input. image should be a 2D or 3D NumPy array.")

    return image_copy

def binarize_all(image_np_arrays):
    """
    This method applies binarization to all the images in the dataset and saves the result in the output directory
    """
    current_directory = os.getcwd()
    output_directory = "output"
    output_directory = os.path.join(current_directory, output_directory)

    counter = 1
    for image in image_np_arrays:
        result = binarize_image(image, 127) # threshold is set to 127
        output_path = os.path.join(output_directory, f"binarized_{counter}.png")
        cv2.imwrite(output_path, result) # store the binarized image into the output directory
        counter = counter + 1

def plot_original_and_binarize(image_np_arrays, number: int):
    """
    This method plots both the original image and the binarized image side by side
    """
    current_directory = os.getcwd()
    output_directory = "output"
    output_directory = os.path.join(current_directory, output_directory)
    dataset_directory = os.path.join(current_directory, "dataset")

    random_ints_list = [random.randint(0, 99) for _ in range(number)]
    if (number < 1 or number > 100):
        print("Plot Error")
        return None
    
    counter = 1
    plt.figure(figsize=(8 * number, 8))
    for i in random_ints_list:
        original = cv2.imread(dataset_directory + "/" + str(i) + ".png")
        binarized = cv2.imread(output_directory + "/binarized_" + str(i) + ".png")

        plt.subplot(2, number, counter * 2 -1)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(2, number, counter * 2)
        plt.imshow(binarized)
        plt.title("Binarized Image")
        plt.axis('off')
        counter = counter + 1

    plt.show()


def main():
    images, image_directory = load_images()
    image_np_arrays = image_to_array(images, image_directory) # each index of this array contains an image in np_array format
    binarize_all(image_np_arrays)
    plot_original_and_binarize(image_np_arrays, number=2)


if __name__ == "__main__":
    main()