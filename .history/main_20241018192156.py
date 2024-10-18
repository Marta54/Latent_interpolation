from image_preprocess import make_square
import matplotlib.pyplot as plt

import glob
import os

from PIL import Image

# Specify the folder path
folder_path = 'C:\\Users\\msro1\\Downloads\\archive (1)\\images'
output_folder = ''
# Use glob to find all .jpg files recursively
jpg_files = glob.glob(os.path.join(folder_path, '**', '*.jpg'), recursive=True)

# Iterate through the found .jpg files
for jpg_file in jpg_files:
    # Perform the desired action on each .jpg file (e.g., resize, compress, etc.)
    print(type(make_square(jpg_file, 512))) 
       # Replace this print statement with your desired action
    break