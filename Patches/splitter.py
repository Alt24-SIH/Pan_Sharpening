import scipy.io as sio
import numpy as np

mat_data = sio.loadmat(r"D:\Downloads\R-PNN-main\R-PNN-main\Patches\RR1_Barcelona.mat")
data_key = 'I_GT'
data_array = mat_data[data_key] 

num_rows, num_cols = data_array.shape[0], data_array.shape[1]
split_size = 40  

row_splits = [(i, min(i + split_size, num_rows)) for i in range(0, num_rows, split_size)]
col_splits = [(j, min(j + split_size, num_cols)) for j in range(0, num_cols, split_size)]

file_idx = 1
for (start_row, end_row) in row_splits:
    for (start_col, end_col) in col_splits:
        split_data = data_array[start_row:end_row, start_col:end_col, :]

        if split_data.shape[0] == split_size and split_data.shape[1] == split_size:
            split_dict = {data_key: split_data}

            output_filename = f'split_data_{file_idx}.mat'
            sio.savemat(output_filename, split_dict)
            print(f"Saved {output_filename} with shape {split_data.shape}")
            file_idx += 1
