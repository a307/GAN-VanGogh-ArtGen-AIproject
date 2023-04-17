from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class ExcelImageDataset(Dataset):
    def __init__(self, excel_path, image_column, transform=None):
        # This will read the excel file and save its content in a dataframe (a 2-D table)
        self.df = pd.read_excel(excel_path)
        # This will get the list of the file paths from our table
        self.file_paths = self.df[image_column].tolist()
        # Make sure that the file paths are correct and complete
        self.file_paths = [os.path.abspath(path) for path in self.file_paths]
        # Save the changes that will be made to the images in the future
        self.transform = transform

    def __len__(self): # This function returns the number of images in our dataset
        return len(self.file_paths)

    def __getitem__(self, index): # This function allows a specific image from the table to be acquired
        image_path = self.file_paths[index] # Find the image filepath via its index
        image = Image.open(image_path).convert('RGB') # Opens the image file and makes sure it has colour
        if self.transform: # If any changes were made apply them
            image = self.transform(image)
        return image # returns the image
