import torch
from model import  Generator
from interpolate_latent import set_seeds, latent_space_interpolation

print('nnn')
# Assuming latent_dim is the dimension of your GAN's latent space, e.g., 100
LATENT_DIM = 100

VERSION = 1
MODEL_PATH = 'C:\\Users\\msro1\\Latent_interpolation\\DCGAN\\pokemon_checkpoints_models\\3\\gen_480.pth'
VIDEO_NAME = 'latent_space_interpolation.avi'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES_GEN = 64
CHANNELS_IMG = 3
Z_DIM = 100
N = 8

FPS = 30
NUM_STEPS = 300
SUPER_RES = True
PLOT = False

# Load your pre-trained model
generator = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
generator.eval()  # Set to evaluation mode

latent_space_interpolation(generator, LATENT_DIM, 
                          num_steps=NUM_STEPS, video_name=VIDEO_NAME, fps=FPS, super_res = SUPER_RES,
                         plot = PLOT)