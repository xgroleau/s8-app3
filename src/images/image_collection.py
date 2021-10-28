import os
import glob
from dataclasses import dataclass


class ImageCollection:
    def __init__(self, base_path=".", filter_name=None):
        # liste de toutes les images
        self.path = glob.glob(fr"{base_path}\*.jpg")
        self.image_folder = fr"{base_path}"
        self.image_list = os.listdir(self.image_folder)
        # Filtrer pour juste garder les images
        self.image_list = [i for i in self.image_list if '.jpg' in i]

        if filter_name:
            self.image_list = list(filter(lambda name: filter_name in name, self.image_list))

