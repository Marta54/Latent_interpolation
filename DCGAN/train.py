import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model2 import Discriminator, Generator, initialize_weights
import matplotlib.pyplot as plt
import os
import random

def set_seeds(seed_value):
    # Python's built-in random module
    random.seed(seed_value)
    torch.manual_seed(seed_value)  # For CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  # For current GPU
        torch.cuda.manual_seed_all(seed_value)  # For all GPUs if using multi-GPU
    # Ensuring deterministic behavior
    torch.backends.cudnn.deterministic = True

SEED = 42
set_seeds(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 2e-4
BATCH_SIZE = 2
IMAGE_SIZE = FEATURES_DISC = FEATURES_GEN = 552

CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 100

VERSION = '4_big'
DATA_FOLDER = 'C:\\Users\\msro1\\Latent_interpolation\\pokemon552'
OUT_FOLDER = f'C:\\Users\\msro1\\Latent_interpolation\\DCGAN\\images_pokemon_train\\{VERSION}'
CHECKPOINT_FOLDER = f'C:\\Users\\msro1\\Latent_interpolation\\DCGAN\\pokemon_checkpoints_models\\{VERSION}'

SAVE_FREQ = 20 # number of epochs after which the model and example images are saved



if not os.path.exists(OUT_FOLDER):
    # Create the folder if it doesn't exist
    os.makedirs(OUT_FOLDER)

if not os.path.exists(CHECKPOINT_FOLDER):
    # Create the folder if it doesn't exist
    os.makedirs(CHECKPOINT_FOLDER)

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),

    ]
)

#dataset = datasets.MNIST(root="dataset/", train = True, transform=transforms, download=True)
dataset = datasets.ImageFolder(root=DATA_FOLDER, transform=transforms)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(8, Z_DIM, 1, 1).to(device)
gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)
        # Train discriminator max log((D(x)) + log(1 - D(G(z))))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph = True)
        opt_disc.step()

        # Train generator min log(1-D(G(z))) -> max log(D(G(Z))) (stabilization)
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1} out of {len(loader)}")
                
    if epoch % SAVE_FREQ == 0:
        torch.save(gen.state_dict(), CHECKPOINT_FOLDER + f'\\gen_{epoch}.pth')
        with torch.no_grad():
            print("Saving examples")
            fake = gen(fixed_noise)
            # take up to 32 examples
            img_grid_real = torchvision.utils.make_grid(
                real[:32], normalize=True
                )
            img_grid_fake = torchvision.utils.make_grid(
                fake[:32], normalize=True
            )
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img_grid_real.permute(1, 2, 0).cpu().numpy())
            plt.title('real images')
            plt.subplot(1, 2, 2)
            plt.imshow(img_grid_fake.permute(1, 2, 0).cpu().numpy())
            plt.title('fake images')
            plt.tight_layout()
            plt.axis('off')
            plt.savefig(OUT_FOLDER + f'\\{epoch}_{batch_idx}.png')
            plt.close()