import torch
import torch.nn as nn

# Search for local GPU (if not available use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define Generator class from Pytorch to generate images
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Define Generator neural network architecture
        self.model = nn.Sequential(
            nn.Linear(100, 256),  # random noise vector (size 100) outputs tensor (size 256)
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 3 * 64 * 64),  # produce 64 x 64 size image
            nn.Tanh()
        )
        # Define the layers
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 3 * 64 * 64) # Change the 256 * 256 for a different size

        # Define the activation functions
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # Apply the first layer and activation
        z = self.fc1(z)
        z = self.leaky_relu(z)

        # Apply the second layer and activation
        z = self.fc2(z)
        z = self.leaky_relu(z)

        # Apply the third layer and activation
        z = self.fc3(z)
        z = self.leaky_relu(z)

        # Apply the fourth layer and activation
        z = self.fc4(z)
        img = self.tanh(z)

        # Reshape the output to the desired image size
        img = img.view(img.size(0), 3, 64, 64) # Change the 256, 256 here for a different size

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


    # Pass randomNoiseVector and reshape it for proper generated image ready to be discriminated
    # def forward(self, randomNoiseVector):
    #     img = self.model(randomNoiseVector).to(device)
    #     img = img.view(img.size(0), 3, 64, 64)
    #     return img
