
import os
import glob


class ImageCollection:
    def __init__(self, filter_name=None):
        # liste de toutes les images
        self.path = glob.glob(r".\baseDeDonneesImages\*.jpg")
        self.image_folder = r".\baseDeDonneesImages"
        self.image_list = os.listdir(self.image_folder)
        # Filtrer pour juste garder les images
        self.image_list = [i for i in self.image_list if '.jpg' in i]

        if filter_name:
            self.image_list = list(filter(lambda name: filter_name in name, self.image_list))
