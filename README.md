# Medical Image Denoising and Enhancement

This project implements various **denoising** and **enhancement techniques** for medical images, particularly focusing on DICOM images. The goal is to remove noise from medical images while preserving and enhancing important features, such as edges and structural details, which are critical for diagnosis.

It uses **Gaussian** and **Median filters** for denoising, and combines them with **low-pass** and **high-pass Fourier filters** to further enhance the resulting images. This is especially useful for tasks like removing noise from X-ray, CT, or MRI scans.

## Features

- **Gaussian and Median Filtering**: Denoising of medical images while balancing between smoothing and edge preservation.
- **Low-pass and High-pass Filtering**: Enhance details (edges) or further smooth the image after denoising.
- **DICOM Image Support**: Specifically tailored for medical imaging formats.
- **Visualization**: Images are visualized using `matplotlib` to observe the results of different filtering techniques.

## Installation

### Prerequisites

1. Python 3.x
2. Install the required libraries:

```bash
pip install numpy matplotlib scipy scikit-image pydicom ipywidgets
```

## Clone the Repository
```bash
git clone https://github.com/your-username/medical-image-denoising.git
cd medical-image-denoising
```

## Usage

The script processes a DICOM medical image and applies different denoising and enhancement techniques.

### How to Run

1. Place your DICOM file in the project folder.
2. Update the file path in the `process_and_enhance_image` function call (replace `'/content/moonesan_3.DCM'` with your file path).
3. Run the script.

```python
process_and_enhance_image('your_image.DCM')
```
This will process the file, apply various denoising and enhancement techniques, and display the output images.

## Output

The script will:

1. Load the original DICOM image and display it.
2. Add Gaussian noise to simulate noisy medical images.
3. Denoise the image using:
   - Gaussian filter
   - Median filter
4. Apply Low-pass and High-pass filtering on the noisy image.
5. Enhance the denoised images by adding back:
   - High-pass filtered image to restore edges and details.
   - Low-pass filtered image to further smooth the image.

### Sample Results

- Original Medical Image
- Noisy Medical Image
- Denoised Images (Gaussian and Median filters)
- Low-pass Filtered Image
- High-pass Filtered Image
- Enhanced Images:
  - Gaussian Denoised + High-Pass
  - Gaussian Denoised + Low-Pass
  - Median Denoised + High-Pass
  - Median Denoised + Low-Pass

## Code Overview

### Main Functions

- `load_dicom_image(file_path)`: Loads and reads a DICOM image.
- `add_gaussian_noise(img, sigma)`: Adds Gaussian noise to simulate noisy medical images.
- `apply_gaussian_filter(img, sigma)`: Applies Gaussian filtering for denoising.
- `apply_median_filter(img, size)`: Applies Median filtering for denoising.
- `ideal_low_filter(radius, img)`: Performs low-pass Fourier filtering.
- `ideal_high_filter(radius, img)`: Performs high-pass Fourier filtering.
- `process_and_enhance_image(file_path)`: Main function to process, denoise, and enhance the medical image.

### Example

If you have a DICOM file called `example.dcm`, you can simply run the following:

```python
process_and_enhance_image('example.dcm')
```
This will process the file, apply various denoising and enhancement techniques, and display the output images.

## Future Enhancements

- Add more advanced denoising techniques (e.g., Wavelet Denoising, Bilateral Filtering).
- Incorporate machine learning models for noise reduction.
- Implement a GUI for easier interaction.

## License

This project is licensed under the [MIT License](LICENSE).
