import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

def visualize_mat_file(file_path):
    """
    Visualizes the content of a .mat file.

    Parameters:
    - file_path (str): Path to the .mat file.
    """
    # Load the .mat file
    mat_data = sio.loadmat(file_path)
    
    # Exclude internal MATLAB keys
    keys = [key for key in mat_data.keys() if not key.startswith('__')]

    print(f"Variables in '{file_path}':")
    for key in keys:
        print(f"- {key}: {mat_data[key].shape if hasattr(mat_data[key], 'shape') else type(mat_data[key])}")

    # Visualize spatial data
    for key in keys:
        data = mat_data[key]
        if isinstance(data, np.ndarray):
            if data.ndim == 2:  # Grayscale image
                plt.figure()
                plt.title(f"{key} (Grayscale)")
                plt.imshow(data, cmap='gray')
                plt.colorbar()
                plt.show()
            elif data.ndim == 3:  # Multispectral image
                plt.figure()
                plt.title(f"{key} (Multispectral - First Band)")
                plt.imshow(data[:, :, 0], cmap='viridis')
                plt.colorbar()
                plt.show()
            else:
                print(f"{key} has unsupported dimensions for visualization: {data.shape}")
    
# File path
file_path1 = 'D:/Downloads/R-PNN-main/R-PNN-main/Output_folder/R-PNN/patch_0_R-PNN.mat'
file_path2 = 'D:/Downloads/R-PNN-main/R-PNN-main/Output_folder/R-PNN/patch_0.mat'
# Visualize the .mat file
visualize_mat_file(file_path1)
visualize_mat_file(file_path2)
