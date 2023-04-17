# import torch.nn as nn
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(3 * 256 * 256, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
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

# New code here
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Define the first linear layer, activation function, and dropout layer
        self.layer1 = nn.Linear(3 * 256 * 256, 1024) # Change the 256 * 256 here for a different size
        self.activation1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(0.3)

        # Define the second linear layer, activation function, and dropout layer
        self.layer2 = nn.Linear(1024, 512)
        self.activation2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(0.3)

        # Define the third linear layer, activation function, and dropout layer
        self.layer3 = nn.Linear(512, 256)
        self.activation3 = nn.LeakyReLU(0.2)
        self.dropout3 = nn.Dropout(0.3)

        # Define the fourth linear layer and its activation function (output layer)
        self.layer4 = nn.Linear(256, 1)
        self.activation4 = nn.Sigmoid()

    def forward(self, img):
        # Reshape the input image to a 1D tensor
        img = img.view(img.size(0), -1)

        # Pass the image through the first layer, activation function, and dropout layer
        img = self.layer1(img)
        img = self.activation1(img)
        img = self.dropout1(img)

        # Pass the image through the second layer, activation function, and dropout layer
        img = self.layer2(img)
        img = self.activation2(img)
        img = self.dropout2(img)

        # Pass the image through the third layer, activation function, and dropout layer
        img = self.layer3(img)
        img = self.activation3(img)
        img = self.dropout3(img)

        # Pass the image through the fourth (output) layer and its activation function
        img = self.layer4(img)
        validity = self.activation4(img)

        # Return the validity score (probability of the image being real)
        return validity