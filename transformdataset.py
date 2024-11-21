import scipy.io
import numpy as np
import skimage.transform

# Load your .mat file (adjust path as needed)
data = scipy.io.loadmat('Sandiego.mat')
sdata = data['Sandiego']

# Function to split the dataset into smaller patches
def split_into_patches(data, patch_size=(64, 64)):
    height, width, bands = data.shape
    patch_height, patch_width = patch_size
    patches = []

    # Iterate over the dataset in steps of patch_size
    for i in range(0, height, patch_height):
        for j in range(0, width, patch_width):
            patch = data[i:i + patch_height, j:j + patch_width, :]
            if patch.shape[0] == patch_height and patch.shape[1] == patch_width:
                patches.append(patch)

    return patches

# Function to transform a single patch
def create_full_sandiego_test_dataset(input_data, ratio=1):
    height, width, bands = input_data.shape
    
    # Generate I_MS_LR (Original Hyperspectral Stack)
    I_MS_LR = input_data

    # Generate I_MS (Upsampled version of original Hyperspectral Stack)
    HR = height * ratio
    WR = width * ratio
    I_MS = np.stack([skimage.transform.resize(I_MS_LR[:, :, i], (HR, WR), mode='reflect', anti_aliasing=True) 
                    for i in range(bands)], axis=2)

    # Generate I_PAN (Panchromatic band)
    I_PAN = np.mean(I_MS, axis=2)

    # Generate Wavelengths array (example values)
    Wavelengths = np.linspace(370, 2510, bands).reshape(bands, 1)

    # Additional parameters to match RR1_Barcelona dataset structure
    L = 16  
    Qblocks_size = 32 
    dim_cut = 11  
    flag_cut_bounds = 1
    printEPS = 0 
    ratio_value = 6  
    sensor = "Prisma"  
    thvalues = 1 

    # Prepare the output dataset
    transformed_dataset = {
        'I_MS_LR': I_MS_LR,
        'I_MS': I_MS,
        'I_PAN': I_PAN,
        'Wavelengths': Wavelengths,
        'I_GT': input_data,  
        'L': L,
        'Qblocks_size': Qblocks_size,
        'dim_cut': dim_cut,
        'flag_cut_bounds': flag_cut_bounds,
        'printEPS': printEPS,
        'ratio': ratio_value,
        'sensor': sensor,
        'thvalues': thvalues
    }
    
    return transformed_dataset

# Split data into patches of size 50x50 (you can adjust the size as needed)
patches = split_into_patches(sdata, patch_size=(64, 64))

# Process and save each patch
for idx, patch in enumerate(patches):
    transformed_patch = create_full_sandiego_test_dataset(patch)
    filename = f'Transformed_Sandiego_Patch_{idx}.mat'
    scipy.io.savemat(filename, transformed_patch)