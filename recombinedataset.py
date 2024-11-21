import os
from scipy.io import loadmat, savemat
import numpy as np

def recombine_patches(input_dir, output_file, original_shape, patch_size, overlap=0):
    """
    Recombines patches saved as .mat files into the original dataset structure,
    while preserving specified values as constants.

    Parameters:
    - input_dir (str): Directory containing the patch .mat files.
    - output_file (str): Path to save the recombined .mat file.
    - original_shape (tuple): (height, width, depth) of the original data.
    - patch_size (tuple): (height, width) of the patches.
    - overlap (int): Overlap in pixels between adjacent patches.
    """
    patch_h, patch_w = patch_size
    stride_h = patch_h - overlap
    stride_w = patch_w - overlap
    original_h, original_w = original_shape[:2]

    # Variables to keep the same as the original
    constant_vars = [
        'Wavelengths', 'L', 'Qblocks_size', 'dim_cut', 
        'flag_cut_bounds', 'printEPS', 'ratio', 'sensor', 'thvalues'
    ]
    constant_values = {}

    # Initialize arrays to store recombined data
    combined_data = {}
    weight_matrix_2d = np.zeros((original_h, original_w), dtype=np.float32)  # Weight matrix for 2D data
    weight_matrix_3d = np.zeros((original_h, original_w, original_shape[2]), dtype=np.float32)  # Weight matrix for 3D data

    # Get the list of patch files
    patch_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mat')])

    # Load constant values from the first patch file
    if patch_files:
        first_patch_file = os.path.join(input_dir, patch_files[0])
        first_patch_data = loadmat(first_patch_file)
        for key in constant_vars:
            if key in first_patch_data:
                constant_values[key] = first_patch_data[key]

    # Loop through patches and recombine
    patch_index = 0
    for i in range(0, original_h - patch_h + 1, stride_h):
        for j in range(0, original_w - patch_w + 1, stride_w):
            # Load the patch data
            patch_file = os.path.join(input_dir, f"Transformed_Sandiego_Patch_{patch_index}.mat")
            patch_data = loadmat(patch_file)

            for key, value in patch_data.items():
                if key.startswith('__') or key in constant_vars:
                    continue  # Skip meta-variables and constant variables

                # Skip variables that do not match expected dimensions
                if value.ndim != 2 and value.ndim != 3:
                    print(f"Skipping variable '{key}' in file '{patch_file}' due to unexpected dimensions: {value.shape}")
                    continue

                if key not in combined_data:
                    # Initialize combined data array for this key
                    combined_shape = list(original_shape)
                    if value.ndim == 3:
                        combined_shape[2] = value.shape[2]  # Preserve depth for 3D data
                    elif value.ndim == 2:
                        combined_shape = original_shape[:2]  # Initialize for 2D data
                    combined_data[key] = np.zeros(combined_shape, dtype=np.float32)  # Use float32 for safe division

                # Place the patch in the combined data array
                try:
                    if value.ndim == 2:  # 2D data (e.g., I_PAN)
                        combined_data[key][i:i+patch_h, j:j+patch_w] += value
                        weight_matrix_2d[i:i+patch_h, j:j+patch_w] += 1
                    elif value.ndim == 3:  # 3D data (e.g., I_MS)
                        combined_data[key][i:i+patch_h, j:j+patch_w, :] += value
                        weight_matrix_3d[i:i+patch_h, j:j+patch_w, :] += 1
                except ValueError as e:
                    print(f"Error combining variable '{key}' from file '{patch_file}': {e}. Skipping this variable.")

            patch_index += 1

    # Normalize the combined data by the weight matrix to handle overlapping regions
    for key in combined_data:
        if combined_data[key].ndim == 2:
            combined_data[key] /= np.maximum(weight_matrix_2d, 1)
        elif combined_data[key].ndim == 3:
            for k in range(combined_data[key].shape[2]):
                combined_data[key][:, :, k] /= np.maximum(weight_matrix_3d[:, :, k], 1)

        # Convert data back to original dtype if needed (optional)
        combined_data[key] = combined_data[key].astype(np.uint16)  # Convert to uint16 if required

    # Add constant values to the combined data
    for key, value in constant_values.items():
        combined_data[key] = value

    # Save the combined data to a new .mat file
    savemat(output_file, combined_data)
    print(f"Recombined data saved to '{output_file}'.")

# Parameters
input_directory = 'D:/Downloads/R-PNN-main/R-PNN-main/Transformed Dataset'
output_file_path = 'D:/Downloads/R-PNN-main/R-PNN-main/Combined_Barcelona.mat'
original_data_shape = (400, 400, 224)  # Replace with the original shape of your data (height, width, depth)
patch_dimensions = (64, 64)  # Patch size (height, width)
overlap_pixels = 0  # Overlap between patches

# Recombine patches
recombine_patches(input_directory, output_file_path, original_data_shape, patch_dimensions, overlap_pixels)
