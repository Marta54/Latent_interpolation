import torch
import torchvision.utils as vutils
import numpy as np
import cv2
from torch.autograd import Variable
from model import  Generator
import matplotlib.pyplot as plt

VERSION = 1
MODEL_PATH = 'C:\\Users\\msro1\\Latent_interpolation\\DCGAN\\gen_500.pth'
VIDEO_NAME = 'latent_space_interpolation.avi'
FPS = 30
NUM_STEPS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES_GEN = 64
CHANNELS_IMG = 3
Z_DIM = 100
N = 8


def set_seeds(seed_value):
    # Python's built-in random module
    torch.manual_seed(seed_value)  # For CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  # For current GPU
        torch.cuda.manual_seed_all(seed_value)  # For all GPUs if using multi-GPU
    # Ensuring deterministic behavior
    torch.backends.cudnn.deterministic = True

SEED = 42
set_seeds(SEED)

# Load your pre-trained model
generator = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
generator.eval()  # Set to evaluation mode

def interpolate_latents(latent_dim, num_steps=60):
    z_start = torch.randn(N, latent_dim, 1, 1) # Initial latent vector
    z_end = torch.randn(N, latent_dim, 1, 1) # Final latent vector
    # Interpolate between the two vectors
    z_interpolates = [(1 - t) * z_start + t * z_end for t in np.linspace(0, 1, num_steps)]
    return z_interpolates

def generate_images(generator, z_interpolates):
    images = []
    with torch.no_grad():  # No need to compute gradients
        for z in z_interpolates:
            z = Variable(z).to(DEVICE)  # Ensure the vector is on CPU
            fake_image = generator(z)[0].permute(1,2,0).detach().cpu()  # Generate the image
            fake_image = (fake_image + 1) / 2.0  # Convert the images back to range [0, 1] from [-1, 1]
            #fake_image = vutils.make_grid(fake_image, normalize=True, scale_each=True)  # Normalize
            images.append(fake_image)
    return images

def plot_images(images, num_cols=5):
    num_images = len(images)
    num_rows = num_images // num_cols + (num_images % num_cols > 0)

    plt.figure(figsize=(num_cols * 2, num_rows * 2))
    for i in range(num_images):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i].numpy(), aspect='auto')  # HxWxC format
        plt.axis('off')  # Hide the axes
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.show()  # Display the plot

def create_video(images, video_name='latent_interpolation.mp4', fps=30):
    height, width = images[0].shape[0], images[0].shape[1]
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    for img_tensor in images:
        img_numpy = img_tensor.numpy()  # Convert to HxWxC format
        img_numpy = (img_numpy * 255).astype(np.uint8)  # Convert to uint8 format
        video.write(img_numpy)

    video.release()
    print(f"Video saved as {video_name}")


def latent_space_interpolation(generator, latent_dim, num_steps=60, video_name='latent_interpolation.mp4', fps=30, plot = None):
    # Step 1: Interpolate latent vectors
    z_interpolates = interpolate_latents(latent_dim, num_steps)

    # Step 2: Generate images from the latent vectors
    images = generate_images(generator, z_interpolates)

    # Step 3: Create a video from the generated images
    create_video(images, video_name=video_name, fps = fps)

    if plot:
        plot_images(images, num_cols=5)

# Assuming latent_dim is the dimension of your GAN's latent space, e.g., 100
latent_dim = 100
latent_space_interpolation(generator, latent_dim, num_steps=NUM_STEPS, video_name=VIDEO_NAME, fps=FPS)
