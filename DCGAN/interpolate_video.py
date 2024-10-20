import torch
import torchvision.utils as vutils
import numpy as np
import cv2
from torch.autograd import Variable


MODEL_PATH = 'C:\\Users\\msro1\\Latent_interpolation\\DCGAN\\pokemon_checkpoints_models\\{VERSION}\\gen_150.pth'
VIDEO_NAME = 'latent_space_interpolation.avi'
FPS = 15
NUM_STEPS = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace with your actual Generator class
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define your architecture

    def forward(self, x):
        # Forward pass through the network
        return x

# Load your pre-trained model
generator = Generator()
generator.load_state_dict(torch.load(MODEL_PATH))
generator.eval()  # Set to evaluation mode

def interpolate_latents(latent_dim, num_steps=60):
    z_start = torch.randn(1, latent_dim).to(DEVICE)  # Initial latent vector
    z_end = torch.randn(1, latent_dim).to(DEVICE)  # Final latent vector

    # Interpolate between the two vectors
    z_interpolates = [(1 - t) * z_start + t * z_end for t in np.linspace(0, 1, num_steps)]
    return z_interpolates

def generate_images(generator, z_interpolates):
    images = []
    with torch.no_grad():  # No need to compute gradients
        for z in z_interpolates:
            z = Variable(z).to('cpu')  # Ensure the vector is on CPU
            fake_image = generator(z).detach().cpu()  # Generate the image
            fake_image = (fake_image + 1) / 2.0  # Convert the images back to range [0, 1] from [-1, 1]
            fake_image = vutils.make_grid(fake_image, normalize=True, scale_each=True)  # Normalize
            images.append(fake_image)
    return images

def create_video(images, video_name='latent_interpolation.mp4', fps=30):
    height, width = images[0].shape[0], images[0].shape[1]
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    for img_tensor in images:
        img_numpy = img_tensor.numpy().transpose(1, 2, 0)  # Convert to HxWxC format
        img_numpy = (img_numpy * 255).astype(np.uint8)  # Convert to uint8 format
        video.write(img_numpy)

    video.release()
    print(f"Video saved as {video_name}")


def latent_space_interpolation(generator, latent_dim, num_steps=60, video_name='latent_interpolation.mp4', fps=30):
    # Step 1: Interpolate latent vectors
    z_interpolates = interpolate_latents(latent_dim, num_steps)

    # Step 2: Generate images from the latent vectors
    images = generate_images(generator, z_interpolates)

    # Step 3: Create a video from the generated images
    create_video(images, video_name=video_name)

# Assuming latent_dim is the dimension of your GAN's latent space, e.g., 100
latent_dim = 100
latent_space_interpolation(generator, latent_dim, num_steps=NUM_STEPS, video_name=VIDEO_NAME, fps=FPS)
