from Discriminator import Discriminator
from Generator import Generator
from Dataset import ExcelImageDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import datetime


# Initialize the networks
generator = Generator()
discriminator = Discriminator()

# Define the loss functions and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load the dataset using the ExcelImageDataset class
dataset = ExcelImageDataset('C:\\Users\\sburk\\PycharmProjects\\archive\\VanGoghPaintings.xlsx', 'image_path', transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

# Define the dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # og batch size 32

image_count = 0

time1 = datetime.datetime.now()
prev_time = datetime.datetime.now()

# set number of epochs system will run
num_epoch = 200;

# Training loop
for epoch in range(num_epoch): # 200 og number of epochs
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
            print(f"[Epoch {epoch}/{num_epoch}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Save generated images
            # Save generated images
            # if epoch % 20 == 0 and i == 0:
            if i % 100 == 0:
                time = datetime.datetime.now()
                current_time = time.strftime("%Y-%m-%d %H:%M:%S.%f")
                save_image(fake_images.data[:25],
                           f"C:\\Users\\sburk\\PycharmProjects\\GAN-artgen-vangogh-AIproject\\image_output\\epoch_{epoch}_image_{image_count}.png",
                           nrow=5, normalize=True)
                image_count += 1;
                print(f"Image created at [Time: {current_time}]")
                if epoch != 0:
                    prev_time = time - prev_time
                    # print(f"Difference from previous: {prev_time}")
                    minutes_diff = str(prev_time.total_seconds() // 60)
                    minutes = float(minutes_diff) * 60
                    seconds_diff = str(prev_time.seconds - minutes)
                    output = "Time elapsed: " + minutes_diff + " minutes, " + seconds_diff + " seconds, " + str(
                        prev_time.microseconds) + " microseconds"
                    print(output)
                prev_time = time