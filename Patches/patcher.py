import os
from scipy.io import loadmat, savemat
import numpy as np

def extract_patches_with_metadata(mat_file, output_dir, patch_size, overlap=0):
    """
    Extracts spatial patches while preserving all specified values (spatial and non-spatial) in all patches.

    Parameters:
    - mat_file (str): Path to the input .mat file.
    - output_dir (str): Directory to save the extracted patches.
    - patch_size (tuple): (height, width) of the patches.
    - overlap (int): Overlap in pixels between adjacent patches.
    """
    # Load the .mat file
    data = loadmat(mat_file)

    # Define keys for spatial data (matrices) and constant metadata
    spatial_keys = ['I_GT', 'I_MS', 'I_MS_LR', 'I_PAN']
    constant_keys = [
        'dim_cut', 'flag_cut_bounds', 'L', 'printEPS', 
        'Qblocks_size', 'ratio', 'sensor', 'thvalues', 'Wavelengths'
    ]

    # Check if keys are present in the loaded data
    spatial_keys = [key for key in spatial_keys if key in data]
    constant_keys = [key for key in constant_keys if key in data]

    # Assume all spatial data have the same height and width (if not, additional handling is needed)
    example_shape = data[spatial_keys[0]].shape
    h, w = example_shape[:2]  # Spatial dimensions

    # Prepare the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Patch dimensions
    patch_h, patch_w = patch_size
    stride_h = patch_h - overlap
    stride_w = patch_w - overlap

    # Extract patches
    patch_count = 0
    for i in range(0, h - patch_h + 1, stride_h):
        for j in range(0, w - patch_w + 1, stride_w):
            patch_data = {}

            # Extract spatial patches
            for key in spatial_keys:
                if key in data and hasattr(data[key], 'shape'):
                    matrix = data[key]
                    if len(matrix.shape) == 2:  # 2D (e.g., I_PAN)
                        patch = matrix[i:i+patch_h, j:j+patch_w]
                    elif len(matrix.shape) == 3:  # 3D (e.g., I_MS)
                        patch_h_dim, patch_w_dim, _ = matrix.shape
                        # Ensure patch dimensions match the spatial data
                        if patch_h <= patch_h_dim and patch_w <= patch_w_dim:
                            patch = matrix[i:i+patch_h, j:j+patch_w, :]
                        else:
                            patch = None  # Skip extraction if dimensions do not match
                    else:
                        patch = None  # Skip unsupported data shapes

                    if patch is not None:
                        patch_data[key] = patch

            # Add constant metadata unchanged to every patch
            for key in constant_keys:
                if key in data:
                    patch_data[key] = data[key]

            # Save the patch if it contains spatial data
            if patch_data:
                patch_filename = os.path.join(output_dir, f"patch_{patch_count}.mat")
                savemat(patch_filename, patch_data)
                patch_count += 1

    print(f"Extracted {patch_count} patches and saved in '{output_dir}'.")

# Parameters
input_file = 'D:/Downloads/R-PNN-main/R-PNN-main/Patches/output_downsampled_data.mat'
output_directory = 'D:/Downloads/R-PNN-main/R-PNN-main/Patches'
patch_dimensions = (64, 64)  # Patch size (height, width)
overlap_pixels = 0  # Overlap between patches

# Extract patches
extract_patches_with_metadata(input_file, output_directory, patch_dimensions, overlap_pixels)
