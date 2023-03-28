import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import pandas as pd
import os
import datetime

# Define the generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 3 * 64 * 64),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 64, 64)
        return img


# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3 * 64 * 64, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        validity = self.model(img)
        return validity

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(100, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, 3 * 32 * 32),  # 32x32 output image size
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.size(0), 3, 32, 32)  # reshape to 3-channel image
#         return img
#
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(3 * 32 * 32, 512),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, img):
#         img = img.view(img.size(0), -1)
#         validity = self.model(img)
#         return validity

# Initialize the networks
generator = Generator()
discriminator = Discriminator()

# Define the loss functions and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

class ExcelImageDataset(Dataset):
    def __init__(self, excel_path, image_column, transform=None):
        self.df = pd.read_excel(excel_path)
        self.file_paths = self.df[image_column].tolist()
        self.file_paths = [os.path.abspath(path) for path in self.file_paths]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image_path = self.file_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# Load the dataset using the ExcelImageDataset class
dataset = ExcelImageDataset('C:\\Users\\sburk\\PycharmProjects\\archive\\VanGoghPaintings.xlsx', 'image_path', transform=transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

# # Load the dataset using the ExcelImageDataset class
# dataset = ExcelImageDataset('C:\\Users\\sburk\\PycharmProjects\\archive\\VanGoghPaintings.xlsx', 'image_path', transform=transforms.Compose([
#     transforms.Resize(32),
#     transforms.CenterCrop(32),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ]))

# Define the dataloader (Batch = 32 images taken. Shuffle means taken at random or not)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # og batch size 32

image_count = 0

time1 = datetime.datetime.now()
prev_time = datetime.datetime.now()

# Training loop
for epoch in range(100): # 200 og number of epochs
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
            print(f"[Epoch {epoch}/{100}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

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
                output = "Time elapsed: "+ minutes_diff + " minutes, " + seconds_diff + " seconds, " + str(
                    prev_time.microseconds) + " microseconds"
                print(output)
            prev_time = time

            # save_image(fake_images.data[:25], f"C:\\Users\\sburk\\PycharmProjects\\GAN-artgen-vangogh-AIproject\\image_output\\{epoch}.png", nrow=5, normalize=True)
# # Training loop with image loop retaining
# generated_images = []
# training_data = []
# image_names = []
# num_epochs = 200
#
# for epoch in range(num_epochs):
#     for i, imgs in enumerate(dataloader):
#         # Adversarial ground truths
#         valid = torch.ones(imgs.size(0), 1)
#         fake = torch.zeros(imgs.size(0), 1)
#
#         # Configure input
#         real_imgs = imgs
#
#         # -----------------
#         # Train Generator
#         # -----------------
#
#         optimizer_G.zero_grad()
#
#         # Sample noise
#         z = torch.randn(imgs.shape[0], 100)
#
#         # Generate a batch of images
#         gen_imgs = generator(z)
#
#         # Loss measures generator's ability to fool the discriminator
#         g_loss = adversarial_loss(discriminator(gen_imgs), valid)
#
#         g_loss.backward()
#         optimizer_G.step()
#
#         # ---------------------
#         # Train Discriminator
#         # ---------------------
#
#         optimizer_D.zero_grad()
#
#         # Measure discriminator's ability to classify real from generated samples
#         real_loss = adversarial_loss(discriminator(real_imgs), valid)
#         fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
#         d_loss = (real_loss + fake_loss) / 2
#
#         d_loss.backward()
#         optimizer_D.step()
#
#         # Append the training data
#         training_data.append(imgs)
#
#         # Print training progress
#         print(f"[Epoch {epoch + 1}/{num_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
#
#         # Save generated images for this epoch
#         if (epoch + 1) % 5 == 0:
#             with torch.no_grad():
#                 z = torch.randn(32, 100)
#                 gen_imgs = generator(z)
#                 generated_images.append(gen_imgs)
#
#                 # Save generated images as PNG files
#                 for j, gen_img in enumerate(gen_imgs):
#                     file_name = f"epoch_{epoch + 1}_image_{j + 1}.png"
#                     file_name = f"C:\\Users\\sburk\\PycharmProjects\\GAN-artgen-vangogh-AIproject\\image_output\\epoch_{epoch + 1}_image_{j + 1}.png"
#                     save_image((gen_img * 0.5) + 0.5, file_name)
#                     image_names.append(file_name)
