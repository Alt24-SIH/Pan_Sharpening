import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load the first .mat file and display the first image
file_path1 = r"D:\Downloads\R-PNN-main\R-PNN-main\output_downsampled_data.mat" 
mat_data1 = scipy.io.loadmat(file_path1)

print(mat_data1.keys())

hyperspectral_data1 = mat_data1['I_MS']  
print(f"Data shape for first image: {hyperspectral_data1.shape}")  

# Select RGB bands for the first image (adjust as needed)
rgb_bands1 = [30, 19, 11]  
rgb_image1 = np.stack([hyperspectral_data1[:, :, rgb_bands1[i]] for i in range(3)], axis=-1)

# Normalize the first image to [0, 1]
rgb_image1 = (rgb_image1 - rgb_image1.min()) / (rgb_image1.max() - rgb_image1.min())

# Increase brightness (apply scaling factor, e.g., 1.5 for 50% increase)
brightness_factor = 2
rgb_image1 = np.clip(rgb_image1 * brightness_factor, 0, 1)  # Ensure values remain within [0, 1]

# Load the second .mat file and display the second image
file_path2 = r"D:\Downloads\R-PNN-main\R-PNN-main\Output_folder\R-PNN\output_downsampled_data_R-PNN.mat"  # Example second file
mat_data2 = scipy.io.loadmat(file_path2)

print(mat_data2.keys())

hyperspectral_data2 = mat_data2['I_MS']  # Replace with appropriate variable name if different
print(f"Data shape for second image: {hyperspectral_data2.shape}")  

# Select RGB bands for the second image (adjust as needed)
rgb_bands2 = [30, 19, 11]  
rgb_image2 = np.stack([hyperspectral_data2[:, :, rgb_bands2[i]] for i in range(3)], axis=-1)

# Normalize the second image to [0, 1]
rgb_image2 = (rgb_image2 - rgb_image2.min()) / (rgb_image2.max() - rgb_image2.min())

# Increase brightness (apply scaling factor)
rgb_image2 = np.clip(rgb_image2 * brightness_factor, 0, 1)  # Ensure values remain within [0, 1]

# Create subplots to display both images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Display the first image
axes[0].imshow(rgb_image1)
axes[0].set_title("First RGB Composite")
axes[0].axis('off')  # Turn off axis for better visualization

# Display the second image
axes[1].imshow(rgb_image2)
axes[1].set_title("Second RGB Composite")
axes[1].axis('off')  # Turn off axis for better visualization

plt.tight_layout()
plt.show()
