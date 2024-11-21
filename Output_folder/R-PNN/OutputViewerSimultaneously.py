import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

def create_rgb_composite(data, rgb_bands=(0, 1, 2)):
    """
    Creates an RGB composite from a multispectral image by selecting three bands.

    Parameters:
    - data (numpy array): Multispectral image with shape (Height, Width, Bands).
    - rgb_bands (tuple): Indices of the bands to use for the RGB composite (default is (0, 1, 2)).

    Returns:
    - RGB composite image (normalized).
    """
    # Ensure bands are within range
    if max(rgb_bands) >= data.shape[2] or min(rgb_bands) < 0:
        raise ValueError("RGB bands are out of range for the provided data.")

    # Select bands for RGB
    rgb_image = data[:, :, list(rgb_bands)].astype(np.float32)

    # Normalize each band individually to [0, 1]
    for i in range(3):
        band = rgb_image[:, :, i]
        band_min, band_max = np.min(band), np.max(band)
        rgb_image[:, :, i] = (band - band_min) / (band_max - band_min + 1e-5)  # Adding a small epsilon to avoid division by zero

    return rgb_image

def visualize_multispectral_data(file_path1, file_path2):
    """
    Visualizes selected bands or composite images from two multispectral .mat files.

    Parameters:
    - file_path1 (str): Path to the first .mat file.
    - file_path2 (str): Path to the second .mat file.
    """
    # Load the .mat files
    mat_data1 = sio.loadmat(file_path1)
    mat_data2 = sio.loadmat(file_path2)
    
    # Select the variables to visualize (ensure these variables exist in the .mat files)
    var_name1 = 'I_GT'  # Update this if needed
    var_name2 = 'I_MS'  # Update this if needed

    if var_name1 not in mat_data1 or var_name2 not in mat_data2:
        print(f"Variables '{var_name1}' or '{var_name2}' not found in the provided files.")
        return

    data1 = mat_data1[var_name1]
    data2 = mat_data2[var_name2]

    print(f"Variable '{var_name1}' from file 1 has shape: {data1.shape}")
    print(f"Variable '{var_name2}' from file 2 has shape: {data2.shape}")

    if data1.ndim == 3 and data2.ndim == 3:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # File 1 visualization as RGB composite
        rgb1 = create_rgb_composite(data1, rgb_bands=(60, 30, 10))  # Adjust RGB bands if necessary
        axes[0].imshow(rgb1)
        axes[0].set_title(f"{var_name1} (RGB Composite) from File 1")
        axes[0].axis('off')

        # File 2 visualization as RGB composite
        rgb2 = create_rgb_composite(data2, rgb_bands=(60, 30, 10))  # Adjust RGB bands if necessary
        axes[1].imshow(rgb2)
        axes[1].set_title(f"{var_name2} (RGB Composite) from File 2")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("Unsupported data format for visualization. Ensure data dimensions are [Height, Width, Bands].")

# File paths (update these with your actual paths)
file_path1 = 'D:/Downloads/R-PNN-main/R-PNN-main/Output_folder/R-PNN/patch_0.mat'
file_path2 = 'D:/Downloads/R-PNN-main/R-PNN-main/Output_folder/R-PNN/patch_0_R-PNN.mat'

# Visualize the multispectral data
visualize_multispectral_data(file_path1, file_path2)
