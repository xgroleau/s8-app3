import os
import glob
from typing import Dict, List
import pickle as pkl


def export_collection(image_names: Dict[str, List[str]], file_name: str):
    """
    Export a collection if image in a pickle file
    """
    f = open(file_name, "wb")
    pkl.dump(image_names, f)
    f.close()


def load_collection_from_file(file_name:str, images_base_path:str):
    """
    Load a collection of images given a certian name
    """
    f = open(file_name, "rb")
    images_names = pkl.load(f)
    f.close()

    collections = {k: ImageCollection(images_base_path, v) for k, v in images_names.items()}
    return collections


class ImageCollection:
    """
    Class to manage a collection of images
    """
    def __init__(self, base_path=".", image_names=None, filter_name=None):
        # liste de toutes les images
        self.path = glob.glob(fr"{base_path}\*.jpg")
        self.image_folder = fr"{base_path}"

        if image_names is None:
            self.image_list = os.listdir(self.image_folder)
            # Filtrer pour juste garder les images
            self.image_list = [i for i in self.image_list if '.jpg' in i]
        else:
            self.image_list = image_names

        if filter_name:
            self.image_list = list(filter(lambda name: filter_name in name, self.image_list))

