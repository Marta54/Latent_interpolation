from image_preprocess import make_square
import matplotlib.pyplot as plt

import glob
import os

from PIL import Image

# Specify the folder path
folder_path = 'C:\\Users\\msro1\\Downloads\\archive\\kaggle-one-shot-pokemon\\pokemon-a'
output_folder = 'C:\\Users\\msro1\\Latent_interpolation\\pokemon552\\images'
# Use glob to find all .jpg files recursively
jpg_files = glob.glob(os.path.join(folder_path, '**', '*.png'), recursive=True)
size = 552

n = 0
# Iterate through the found .jpg files
for jpg_file in jpg_files:
    make_square(jpg_file, 552).save(os.path.join(output_folder, f'{n}.jpg'))
    n+=1
   