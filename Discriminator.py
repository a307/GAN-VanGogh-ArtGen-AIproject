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

# Define Discriminator class from Pytorch to discriminate generated images via probability of realness
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # First linear layer of network creates specified image and as well as some initial properties
        self.layer1 = nn.Linear(3 * 64 * 64, 1024) # Change the 256 * 256 here for a different size
        self.activation1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(0.3)

        # Second Linear layer define input and output sizes of tensor (1024,512)
        self.layer2 = nn.Linear(1024, 512)
        self.activation2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(0.3)

        # Third layer defines another tensor and sets more properties of image object
        self.layer3 = nn.Linear(512, 256)
        self.activation3 = nn.LeakyReLU(0.2)
        self.dropout3 = nn.Dropout(0.3)

        # Fourth layer activation function determining probability of realness for the generator image
        self.layer4 = nn.Linear(256, 1)
        self.activation4 = nn.Sigmoid()

    # The Forward class takes an image from the generator and returns it's probability of realism in comparison to the
    # Van Gogh database images
    def forward(self, img):
        # Reshape input image from generator to a 1D tensor to prepare for scoring
        img = img.view(img.size(0), -1)

        # Activate first layer of discriminator network by passing image to it
        img = self.layer1(img)
        img = self.activation1(img)
        img = self.dropout1(img)

        # Second layer passing
        img = self.layer2(img)
        img = self.activation2(img)
        img = self.dropout2(img)

        # Third layer passing
        img = self.layer3(img)
        img = self.activation3(img)
        img = self.dropout3(img)

        # Finally pass it to the fourth layer and score its probability
        img = self.layer4(img)
        validity = self.activation4(img)

        # Return the probability to training loop
        return validity
