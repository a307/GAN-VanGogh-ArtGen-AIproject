import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


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


if __name__ == '__main__':
    # Create an instance of the ExcelImageDataset class

    dataset = ExcelImageDataset('C:\\Users\\sburk\\PycharmProjects\\archive\\VanGoghPaintings.xlsx', 'image_path', transform=transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    # Create a PyTorch DataLoader to load the data in batches
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Get a batch of images from the DataLoader
    batch = next(iter(dataloader))

    # Denormalize the images
    images = batch * 0.5 + 0.5

    # Convert the images to a numpy array
    images = images.numpy()

    # Transpose the image array from (batch_size, channels, height, width) to (batch_size, height, width, channels)
    images = images.transpose((0, 2, 3, 1))

    # Visualize the images
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.imshow(images[i])
        ax.axis('off')

    plt.show()
