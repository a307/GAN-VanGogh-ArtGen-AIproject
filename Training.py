from Discriminator import Discriminator
from Generator import Generator
from Dataset import ExcelImageDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image



# Initialize the networks
generator = Generator()
discriminator = Discriminator()

# Define the loss functions and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load the dataset using the ExcelImageDataset class
dataset = ExcelImageDataset('C:\\Users\\Isaiah\\PycharmProjects\\Archive\\VanGoghPaintings.xlsx', 'image_path', transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

# Define the dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # og batch size 32

# Training loop
for epoch in range(5): # 200 og number of epochs
    for i, real_images in enumerate(dataloader):

        # Generate a batch of fake images
        z = torch.randn(real_images.shape[0], 100)
        fake_images = generator(z)

        # Train the discriminator
        optimizer_D.zero_grad()

        real_labels = torch.ones(real_images.shape[0], 1)
        fake_labels = torch.zeros(real_images.shape[0], 1)

        real_loss = adversarial_loss(discriminator(real_images), real_labels)
        fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # Train the generator
        optimizer_G.zero_grad()

        z = torch.randn(real_images.shape[0], 100)
        fake_images = generator(z)

        fake_labels = torch.ones(real_images.shape[0], 1)
        g_loss = adversarial_loss(discriminator(fake_images), fake_labels)

        g_loss.backward()
        optimizer_G.step()

        # Output training progress
        if i % 10 == 0:
            print(f"[Epoch {epoch}/{5}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Save generated images
       # if epoch % 20 == 0 and i == 0:
        if i % 100 == 0:
            save_image(fake_images.data[:10], f"C:\\Users\\Isaiah\\PycharmProjects\\GAN-VanGogh-ArtGen-AIproject\\image_output\\{epoch}.png", nrow=5, normalize=True)