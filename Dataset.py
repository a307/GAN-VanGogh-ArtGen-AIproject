from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

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
