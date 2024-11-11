import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Define the original path to start searching
base_path = r'data'

# A list to store the length (number of dimensions) of each .nii file
nii_lengths = []

# Recursively search through all subfolders for .nii files
for root, _, files in os.walk(base_path):
    # Loop through all files in the current directory
    for file in files:
        # Check if the file has a .nii extension
        if file.endswith(".nii"):
            # Path to the .nii file
            file_path = os.path.join(root, file)
            
            # Load the .nii file
            nii_data = nib.load(file_path)
        
            # Get the shape (dimensions) of the .nii file
            volume_shape = nii_data.shape
        
            # Save the length (number of dimensions) to the list
            nii_lengths.append(volume_shape[2])

print("Lengths of the .nii files:", nii_lengths)

# Calculate percentiles
Q1 = np.percentile(nii_lengths, 25)
Q3 = np.percentile(nii_lengths, 75)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the size as needed

# Separate data based on the percentiles
within_percentiles = [y for y in nii_lengths if Q1 <= y <= Q3]
outside_percentiles = [y for y in nii_lengths if y < Q1 or y > Q3]

# Plot the data
ax.scatter(np.random.normal(1, 0.04, size=len(outside_percentiles)), outside_percentiles, 
           alpha=0.7, edgecolor='#000000', facecolor='#156082', s=40, label='Discarded Volumes')
ax.scatter(np.random.normal(1, 0.04, size=len(within_percentiles)), within_percentiles, 
           alpha=0.9, edgecolor='#000000', facecolor='#003366', s=70, label='Usable Volumes')

# Add labels and title
ax.set_xlabel("NII Files", fontsize=14, weight='bold')
ax.set_ylabel("Transversal Slices Number", fontsize=14, weight='bold')

# Remove the x-axis ticks and labels
ax.set_xticks([])
ax.set_xlim(0.5, 1.5)

# Add lines for 25th and 75th percentiles
ax.axhline(Q1, color='gray', linestyle=':', linewidth=1.25, label='25th Percentile')
ax.axhline(Q3, color='gray', linestyle=':', linewidth=1.25, label='75th Percentile')

# Add a legend
ax.legend()

# Customize the spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Ensure the left and bottom spines are visible
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# Save the plot without the top and right spines
file_name = "nii_scatter_plot_no_outline.png"
plt.savefig(file_name, dpi=500, bbox_inches='tight', pad_inches=0)  # Save with specified DPI and adjust bounding box

# Show the plot
plt.show()
