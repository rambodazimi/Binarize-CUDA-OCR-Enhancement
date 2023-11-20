
# 1. Sequential Binarize

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

# 2. Parallel Binarization

## Introduction

This Python script provides a parallelized image binarization process using the CUDA-enabled CuPy library. The binarization is applied to the same dataset of images used in the sequential part, and the execution time for each image is analyzed and visualized.

## Dependencies

- OpenCV (cv2)
- NumPy
- Matplotlib
- CuPy
- Numba

Make sure to install these dependencies before running the script.

```bash
pip install opencv-python numpy matplotlib cupy numba
```

## Usage

1. Clone the repository

2. Place your dataset of PNG images in the "dataset" directory.

3. Run the script:

```bash
python binarize_parallel.py
```

## Code Explanation

### `load_images()`

This function retrieves and sorts PNG images from the "dataset" directory.

### `image_to_array(images: list, image_directory: str)`

Converts each image to CuPy arrays, representing RGBA values.

### `binarize_image_cupy(image, threshold)`

Applies binarization to a given image in CuPy format based on a specified threshold.

### `binarize_all_parallel(image_cuda_arrays)`

Applies binarization to all images in the dataset in parallel using CuPy and CUDA.

### `plot_execution_time(image_cupy_arrays)`

Plots the execution time of binarization for each image and saves the plot as "analysis/parallel_execution_time.png."

### `main()`

Calls the necessary functions to load images, convert them to CuPy arrays, and plot the execution time.

### `__main__`

Executes the main function when the script is run.

## Results

The script generates a plot showing the execution time for binarizing each image in the dataset and saves it as "analysis/parallel_execution_time.png."

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/) - Computer Vision Library
- [NumPy](https://numpy.org/) - Numerical Computing Library
- [Matplotlib](https://matplotlib.org/) - Plotting and Visualization Library
- [CuPy](https://cupy.dev/) - GPU-accelerated Library for NumPy
- [Numba](https://numba.pydata.org/) - Just-in-time Compilation for Python

Feel free to customize this documentation according to your project's specific details and requirements.


## Output
![test](https://i.ibb.co/vxMtwTb/sequential-execution-time.png)
![test](https://i.ibb.co/bLcKNwG/parallel-execution-time.png)
![test](https://i.ibb.co/PZLfGRG/sequential-binary-comparison.png)
![test](https://i.ibb.co/1dwqyBD/Screenshot-2023-11-20-at-2-49-13-AM.png)

