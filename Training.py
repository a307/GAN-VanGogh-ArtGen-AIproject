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

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Initialize the discriminator and generator network classes as well as link them to CPU/GPU
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# loss function for discriminator prediction scores based on fake and real images from van gogh dataset and generator
LOSS = nn.BCELoss().to(device)

# generator optimizer updates parameters of generator in training loop
gen_opt = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# discriminator optimizer updates parameters of discriminator in training loop
dis_opt = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# get van gogh dataset via ExcelImageDataset class

# dataset = ExcelImageDataset('C:\\Users\\Isaiah\\PycharmProjects\\Archive\\'
#                             'VanGoghPaintingsColour.xlsx', 'image_path', transform = transforms.Compose([

# smaller dataset excluding drawings, sketches in letters, and young van gogh works
dataset = ExcelImageDataset('C:\\Users\\sburk\\PycharmProjects\\archive\\'
                            'VanGoghPaintingsColour.xlsx', 'image_path', transform=transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.RandomHorizontalFlip(),  # 50% change of flip
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

# Initialize dataloader which will act as object for accessing van gogh dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # batch size 32

image_count = 0

# set amount of epoch to be run
epoch_number = 500

# Main Training loop
for epoch in range(epoch_number): # 200 og number of epochs
    for i, real_images in enumerate(dataloader):
        real_images = real_images.to(device)

        # create some fake images using random noise vectors as input into generator to kick it off
        noisey_vec = torch.randn(real_images.shape[0], 100).to(device)
        fake_images = generator(noisey_vec).to(device)

        # DISCRIMINATOR TRAINING----------------------------------

        # clear gradient
        dis_opt.zero_grad()

        # generate tensors filled with real images and another with fake images as well as send them to CPU/GPU
        real_labelImages = torch.ones(real_images.shape[0], 1).to(device)
        fake_labelimages = torch.zeros(real_images.shape[0], 1).to(device)

        # Real image discrimination: gets discriminator classes probability scores for real images from Van Gogh data
        real_image_prediction = discriminator(real_images)

        # Then calculates the real loss of how accurate its predictions were
        realLoss = LOSS(real_image_prediction, real_labelImages)


        # Fake image discrimination: Same as above code but for fake images (ie images made by the generator class)
        fake_image_prediction = discriminator(fake_images.detach())
        fakeLoss = LOSS(fake_image_prediction, fake_labelimages)

        # Calculate total loss for the discriminator based on its ability to score fake and real Van Gogh images
        # Then use the gradients to update the discriminators parameters essentially making it learn and improve
        d_loss = realLoss + fakeLoss
        d_loss.backward()
        dis_opt.step()

        # GENERATOR TRAINING--------------------------------------

        # clear gradient
        gen_opt.zero_grad()

        # generate batch of random noise vectors and send em to the generator for random fake image creation
        noisey_vec = torch.randn(real_images.shape[0], 100).to(device)
        fake_images_generated = generator(noisey_vec).to(device)

        # Now calculate the loss based on discriminators predictions of the fake images compared to real images
        # This causes the generator to 'learn' by figuring out how real looking its fake images are
        g_loss = LOSS(discriminator(fake_images_generated), real_labelImages)

        # use the gradient to go back and update the generators parameters based on it via optimizer
        g_loss.backward()
        gen_opt.step()

        # Output training progress via console display currently outputting every ten images
        if i % 10 == 0:
            print(f"[Epoch {epoch}/{epoch_number}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Save generated images to image_output folder each epoch
        if i % 100 == 0:
            time = datetime.datetime.now()
            current_time = time.strftime("%Y-%m-%d %H:%M:%S.%f")

            # save_image(fake_images.data[:10],
            #            f"C:\\Users\\Isaiah\\PycharmProjects\\GAN-VanGogh-ArtGen-AIproject\\image_output\\"
            #            f"{epoch}.png", nrow=5, normalize=True)

            save_image(fake_images.data[:25],
                       f"C:\\Users\\sburk\\PycharmProjects\\GAN-artgen-vangogh-AIproject\\image_output\\"
                       f"epoch_{epoch}_image_{image_count}.png", nrow=5, normalize=True)
            image_count += 1
            print(f"Image created at [Time: {current_time}]")

            # finally calculate the rate at which each epoch is completed (ie 35 seconds has passed since last epoch)
            if epoch != 0:
                prev_time = time - prev_time
                minutes_diff = str(prev_time.total_seconds() // 60)
                minutes = float(minutes_diff) * 60
                seconds_diff = str(prev_time.seconds - minutes)
                output = "Time elapsed: " + minutes_diff + " minutes, " + seconds_diff + " seconds, " + str(
                    prev_time.microseconds) + " microseconds"
                print(output)
            prev_time = time
