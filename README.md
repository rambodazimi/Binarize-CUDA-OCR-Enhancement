# Sequential Binarize

This Python script performs sequential binarization task on a dataset of images. It includes functions to load images from a specified dataset directory, convert them to NumPy arrays, apply binarization, and plot pairs of original and binarized images.
The dataset can be found at: https://www.kaggle.com/datasets/alizahidraja/captcha-data

## Prerequisites

Make sure you have the following libraries installed:

- `cv2` (OpenCV): Image processing library
- `numpy`: NumPy for array operations
- `matplotlib`: Plotting library
- `random`: Standard Python library for generating random numbers

You can install the required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository:

2. Run the script:

   ```bash
   python binarize_sequential.py
   ```
   
## Directory Structure

1. **dataset**
   - Contains 100 png images from the dataset (before applying the filters)
   - Images are 200x50 pixels
2. **output**
   - Contains the binarized images after running the Python script 

## Functions

1. **load_images()**
   - Description: Iterates over all the files in the dataset directory and retrieves image file names.
   - Output: Tuple containing a list of image file names and the full path to the image directory.

2. **image_to_array(images: list, image_directory: str)**
   - Description: Converts images to NumPy arrays and stores them in a list.
   - Input:
     - `images`: List of image file names.
     - `image_directory`: Full path to the image directory.
   - Output: NumPy array containing images in RGB format.

3. **binarize_image(image: np.array, threshold: int)**
   - Description: Applies binarization with a given threshold to the given image.
   - Input:
     - `image`: NumPy array representing the image.
     - `threshold`: Binarization threshold.
   - Output: Binarized image.
   - The binarization process is applied to each pixel in the image. For RGB images, the function converts each pixel to grayscale intensity using the formula intensity = 0.299R + 0.587G + 0.114B.
   -  Then, it compares the grayscale intensity with the specified threshold and sets the pixel value accordingly.
   -  The result is a binary image where pixels are either set to 255 (white) or 0 (black) based on the grayscale intensity and the specified threshold.

4. **binarize_all(image_np_arrays)**
   - Description: Applies binarization to all the images in the dataset and saves the result in the output directory.
   - Output: Array containing individual execution time in seconds

5. **plot_original_and_binarize(image_np_arrays, number: int)**
   - Description: Plots pairs of original and binarized images for a specified number of randomly selected images.
   - Input:
     - `image_np_arrays`: NumPy array containing images in RGB format.
     - `number`: Number of image pairs to plot
     
6. **plot_execution_time(image_np_arrays)**
   - Description: Plots individual execution time of binarization on selected images
   - Input:
     - `image_np_arrays`: NumPy array containing images in RGB format.

7. **main()**
   - Description: Executes the main processing workflow. Calls functions to load images, convert them to arrays, apply binarization, and plot pairs of images.

## Output
![test](https://i.ibb.co/vxMtwTb/sequential-execution-time.png)
![test](https://i.ibb.co/bLcKNwG/parallel-execution-time.png)
![test](https://i.ibb.co/PZLfGRG/sequential-binary-comparison.png)
![test](https://i.ibb.co/1dwqyBD/Screenshot-2023-11-20-at-2-49-13-AM.png)

