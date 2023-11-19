# ECSE-420-Final-Project

## Dataset
CAPTCHA dataset containing 100 images (png)

https://www.kaggle.com/datasets/alizahidraja/captcha-data/data

## Sequential Binarize

---

This Python script performs basic image processing tasks on a dataset of images. It includes functions to load images from a specified dataset directory, convert them to NumPy arrays, apply binarization, and plot pairs of original and binarized images.

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

   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```

2. Run the script:

   ```bash
   python image_processing_script.py
   ```

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

4. **binarize_all(image_np_arrays)**
   - Description: Applies binarization to all the images in the dataset and saves the result in the output directory.

5. **plot_original_and_binarize(image_np_arrays, number: int)**
   - Description: Plots pairs of original and binarized images for a specified number of randomly selected images.
   - Input:
     - `image_np_arrays`: NumPy array containing images in RGB format.
     - `number`: Number of image pairs to plot.

6. **main()**
   - Description: Executes the main processing workflow. Calls functions to load images, convert them to arrays, apply binarization, and plot pairs of images.

## Usage

```bash
python image_processing_script.py
```

