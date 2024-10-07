# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.filters import median
from skimage.morphology import disk
from numpy.fft import fft2, ifft2, fftshift
import pydicom
import ipywidgets as widgets
from IPython.display import display

# Function to load a DICOM image
def load_dicom_image(file_path):
    dicom_file = pydicom.dcmread(file_path)
    img = dicom_file.pixel_array
    return img

# Function to display an image using matplotlib
def display_image(img, title="Image", cmap='gray'):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to add Gaussian noise to simulate noise in medical images
def add_gaussian_noise(img, mean=0, sigma=0.05):
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy_img = img + gauss
    noisy_img = np.clip(noisy_img, 0, 1)  # Normalize between 0 and 1 for images
    return noisy_img

# Function to apply Gaussian filter to denoise the image
def apply_gaussian_filter(img, sigma=1):
    return gaussian_filter(img, sigma=sigma)

# Function to apply Median filter to denoise the image
def apply_median_filter(img, size=3):
    return median(img, disk(size))

# Fourier-based ideal low-pass filter
def ideal_low_filter(radius, img):
    m, n = img.shape
    cr, cc = m // 2, n // 2
    mask = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if np.sqrt((i - cr) ** 2 + (j - cc) ** 2) <= radius:
                mask[i, j] = 1
    dummy = fftshift(fft2(img))
    dummy = dummy * mask
    return np.abs(ifft2(dummy))

# Fourier-based ideal high-pass filter
def ideal_high_filter(radius, img):
    m, n = img.shape
    cr, cc = m // 2, n // 2
    mask = np.ones((m, n))
    for i in range(m):
        for j in range(n):
            if np.sqrt((i - cr) ** 2 + (j - cc) ** 2) <= radius:
                mask[i, j] = 0
    dummy = fftshift(fft2(img))
    dummy = dummy * mask
    return np.abs(ifft2(dummy))

# Process the medical image with denoising and enhancement
def process_and_enhance_image(file_path):
    # Load and normalize the medical image
    img = load_dicom_image(file_path)
    img = exposure.rescale_intensity(img, out_range=(0, 1))
    display_image(img, title="Original Medical Image")

    # Add noise to the medical image
    noisy_img = add_gaussian_noise(img)
    display_image(noisy_img, title="Noisy Medical Image")

    # Denoising with Gaussian filter
    denoised_gaussian = apply_gaussian_filter(noisy_img, sigma=1)
    display_image(denoised_gaussian, title="Denoised Image (Gaussian Filter)")

    # Denoising with Median filter
    denoised_median = apply_median_filter(noisy_img, size=3)
    display_image(denoised_median, title="Denoised Image (Median Filter)")

    # Apply low-pass filtering to the original noisy image
    low_pass_filtered = ideal_low_filter(20, noisy_img)
    display_image(low_pass_filtered, title="Low-Pass Filtered Image")

    # Apply high-pass filtering to the original noisy image
    high_pass_filtered = ideal_high_filter(20, noisy_img)
    display_image(high_pass_filtered, title="High-Pass Filtered Image")

    # Enhancement 1: Add high-pass filtered image to denoised Gaussian image
    enhanced_with_high_pass_gaussian = denoised_gaussian + high_pass_filtered
    enhanced_with_high_pass_gaussian = np.clip(enhanced_with_high_pass_gaussian, 0, 1)
    display_image(enhanced_with_high_pass_gaussian, title="Enhanced with High-Pass (Gaussian Filter)")

    # Enhancement 2: Add low-pass filtered image to denoised Gaussian image
    enhanced_with_low_pass_gaussian = denoised_gaussian + low_pass_filtered
    enhanced_with_low_pass_gaussian = np.clip(enhanced_with_low_pass_gaussian, 0, 1)
    display_image(enhanced_with_low_pass_gaussian, title="Enhanced with Low-Pass (Gaussian Filter)")

    # Enhancement 3: Add high-pass filtered image to denoised Median image
    enhanced_with_high_pass_median = denoised_median + high_pass_filtered
    enhanced_with_high_pass_median = np.clip(enhanced_with_high_pass_median, 0, 1)
    display_image(enhanced_with_high_pass_median, title="Enhanced with High-Pass (Median Filter)")

    # Enhancement 4: Add low-pass filtered image to denoised Median image
    enhanced_with_low_pass_median = denoised_median + low_pass_filtered
    enhanced_with_low_pass_median = np.clip(enhanced_with_low_pass_median, 0, 1)
    display_image(enhanced_with_low_pass_median, title="Enhanced with Low-Pass (Median Filter)")

# Run the process with your DICOM image
process_and_enhance_image('/content/moonesan_3.DCM')