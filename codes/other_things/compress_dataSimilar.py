import torch
import numpy as np

# Load the .pth file
data = torch.load('data/volumes_[0, 1]_[1600, -600]_masks_Square_0.4.pth',map_location="cpu")

# Assume data is a list of lists of tensors
# Convert the list of lists of tensors to a dictionary of NumPy arrays
np_data = {}
for i, outer_list in enumerate(data):
    for j, tensor in enumerate(outer_list):
        np_data[f'array_{i}_{j}'] = tensor.numpy()

# Save to a compressed .npz file
np.savez_compressed('data/volumes_[0, 1]_[1600, -600]_masks_Square_0.4.npz', **np_data)