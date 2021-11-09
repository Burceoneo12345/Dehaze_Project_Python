"""
file: test_data.py
about: build the test dataset
author: Junhang Wang
date: 26/10/21
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize


# --- Validation/test dataset --- #
class TestData(data.Dataset):
    def __init__(self, test_data_dir):
        super().__init__()
        val_list = test_data_dir + 'test_list.txt'
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
        self.haze_names = haze_names
        self.test_data_dir = test_data_dir

    def get_images(self, index):
        haze_name = self.haze_names[index]
        haze_img = Image.open(self.test_data_dir + 'haze/' + haze_name)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        haze = transform_haze(haze_img)

        return haze, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
