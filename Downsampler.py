import numpy as np
import torch
import scipy.io
from scipy.ndimage import gaussian_filter
from torch.nn.functional import interpolate

def mtf(image, sensor, ratio):
    """
    Applies a Gaussian low-pass filter to simulate the sensor's modulation transfer function (MTF).
    This function smoothens the image to mimic the spatial resolution of the sensor.
    
    Parameters:
    - image: Torch tensor of shape (1, Bands, Height, Width).
    - sensor: String identifier for the sensor (e.g., 'PRISMA').
    - ratio: Downsampling ratio.

    Returns:
    - Torch tensor after applying MTF.
    """
    sigma = ratio / 2.0  # Standard deviation of Gaussian kernel
    numpy_image = image.squeeze(0).numpy()  # Convert to numpy for processing
    filtered_image = np.array([gaussian_filter(band, sigma=sigma) for band in numpy_image])
    return torch.from_numpy(filtered_image[None, :, :, :])

def mtf_pan(image, sensor, ratio):
    """
    Applies a Gaussian low-pass filter to the panchromatic (PAN) image for MTF simulation.
    
    Parameters:
    - image: Torch tensor of shape (1, 1, Height, Width).
    - sensor: String identifier for the sensor (e.g., 'PRISMA').
    - ratio: Downsampling ratio.

    Returns:
    - Torch tensor after applying MTF.
    """
    sigma = ratio / 2.0
    numpy_image = image.squeeze(0).squeeze(0).numpy()
    filtered_image = gaussian_filter(numpy_image, sigma=sigma)
    return torch.from_numpy(filtered_image[None, None, :, :])

def ideal_interpolator(image, ratio):
    """
    Upsamples the hyperspectral image using bilinear interpolation.
    
    Parameters:
    - image: Torch tensor of shape (1, Bands, Height, Width).
    - ratio: Upsampling ratio.

    Returns:
    - Torch tensor after interpolation.
    """
    return interpolate(image, scale_factor=ratio, mode='bilinear', align_corners=False)

def downsample_hs_pan(hs, pan, ratio):
    """
    Downsamples the hyperspectral and PAN data while preserving spectral and spatial integrity.

    Parameters:
    - hs: Hyperspectral image (numpy array of shape Height x Width x Bands).
    - pan: Panchromatic image (numpy array of shape Height x Width).
    - ratio: Downsampling ratio.

    Returns:
    - hs_lr: Low-resolution hyperspectral image.
    - pan_lr: Low-resolution panchromatic image.
    - hs_lr_exp: Expanded low-resolution hyperspectral image.
    """
    # Convert hyperspectral image to torch format
    hs = np.moveaxis(hs, -1, 0)[None, :, :, :].astype(np.float32)
    pan = pan[None, None, :, :].astype(np.float32)

    hs = torch.from_numpy(hs).float()
    pan = torch.from_numpy(pan).float()

    # Apply MTF to simulate sensor effects
    hs_lp = mtf(hs, 'AVIRIS', ratio)
    pan_lp = mtf_pan(pan, 'AVIRIS', ratio)

    # Downsample using nearest neighbor
    hs_lr = torch.nn.functional.interpolate(hs_lp, scale_factor=1 / ratio, mode='nearest-exact')
    pan_lr = torch.nn.functional.interpolate(pan_lp, scale_factor=1 / ratio, mode='nearest-exact')

    # Expand the low-resolution hyperspectral image
    hs_lr_exp = torch.round(torch.clip(ideal_interpolator(hs_lr.double(), ratio), 0, 2**16))

    # Convert back to numpy format
    hs_lr = np.squeeze(hs_lr.numpy()).astype(np.uint16)
    pan_lr = np.squeeze(pan_lr.numpy()).astype(np.uint16)
    hs_lr_exp = np.squeeze(hs_lr_exp.numpy()).astype(np.uint16)

    # Rearrange dimensions
    hs_lr = np.moveaxis(hs_lr, 0, -1)
    hs_lr_exp = np.moveaxis(hs_lr_exp, 0, -1)

    return hs_lr, pan_lr, hs_lr_exp

# Load the AVIRIS dataset
data2 = scipy.io.loadmat('D:/Downloads/R-PNN-main/R-PNN-main/Sandiego.mat')
aviris_image = data2['Sandiego']  # Adjust key based on your dataset structure

# Create the panchromatic image (mean of selected bands)
pan_image = np.mean(aviris_image[:, :, [30, 19, 11]], axis=2)  # Adjust indices based on RGB-like bands
height, width, bands = aviris_image.shape
Wavelengths = np.linspace(370, 2510, bands).reshape(bands, 1)

# Downsampling ratio
ratio = 3  

# Perform downsampling
hs_lr, pan_lr, hs_lr_exp = downsample_hs_pan(aviris_image, pan_image, ratio)

# Save the processed data to a .mat file
output_filename = 'D:/Downloads/R-PNN-main/R-PNN-main/Patches/output_downsampled_data.mat'
scipy.io.savemat(output_filename, {
    'dim_cut': 11,
    'flag_cut_bounds': 1,
    'I_MS_LR': hs_lr,
    'I_MS': hs_lr_exp,
    'I_GT': aviris_image,
    'I_PAN': pan_lr,
    'L': 16,
    'printEPS': 0,
    'ratio': 6,
    'sensor': "AVIRIS",
    'thvalues': 1,
    'Wavelengths': Wavelengths
})

print(f"Data has been saved to {output_filename}")
