import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Will set the device used to the GPU is cuda is working

class Generator(nn.Module):
    def __init__(self):
        # The process for our image generation is initialized
        super(Generator, self).__init__()

        # The first step will be to take 100 random numbers and change them into 256 new numbers
        self.fc1 = nn.Linear(100, 256)

        # The second step will be to transform the previous 256 numbers into 512 new ones
        self.fc2 = nn.Linear(256, 512)

        # A similar thing happens here that happened in the second step
        self.fc3 = nn.Linear(512, 1024)

        # Here a picture will be made with the 1024 numbers from before, it will have colour and be of size 256 pixels by 256 pixels
        self.fc4 = nn.Linear(1024, 3 * 256 * 256) # Change the 256 * 256 here for a different image size

        # The Leaky ReLU function helps the AI learn more efficiently by tweaking the numbers slightly
        self.leaky_relu = nn.LeakyReLU(0.2)

        # This function helps ensure that the image looks more realistic
        self.tanh = nn.Tanh()

    def forward(self, noise_vector): # The forward function helps create an image from a noise vector
        # The noise vector is transformed
        noise_vector = self.fc1(noise_vector)
        # Leaky ReLU makes the vector look better
        noise_vector = self.leaky_relu(noise_vector)

        # The noise vector is transformed for the second time
        noise_vector = self.fc2(noise_vector)
        # Leaky ReLU makes the vector look even better before
        noise_vector = self.leaky_relu(noise_vector)

        # The noise vector is transformed for the third time
        noise_vector = self.fc3(noise_vector)
        # Leaky ReLU makes the vector look even better before
        noise_vector = self.leaky_relu(noise_vector)

        # The noise vector is transformed for the last time
        noise_vector = self.fc4(noise_vector)
        # The Tanh function then makes the vector a "realistic" looking image
        img = self.tanh(noise_vector)

        # The image is resized to 256 pixels by 256 pixels and sent to the GPU
        img = img.view(img.size(0), 3, 256, 256) # Change the 256, 256 here for a different image size
        return img.to(device)

# Old Code
# import torch
# import torch.nn as nn
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
#             nn.Linear(1024, 3 * 256 * 256),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         img = self.model(z).to(device)
#         img = img.view(img.size(0), 3, 256, 256)
#         return img


