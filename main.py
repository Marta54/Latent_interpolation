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

n = 0
# Iterate through the found .jpg files
for jpg_file in jpg_files:
    make_square(jpg_file, 512).save(f'pokemon_square\\{n}.jpg')
    n+=1
   