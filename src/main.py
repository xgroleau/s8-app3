'''
Script for image visualization

'''
import matplotlib.pyplot as plt

from src.images import random_image_selector
from src.images.image_collection import ImageCollection, CategorizedImageCollection
from src.params.extract_param import *
from src.params.param import param_1d, get_images
from src.visualization import images_display
from src.visualization.view_histogram import histogrammes


def main():
    view_hist = False
    coast = ImageCollection(base_path=r"baseDeDonneesImages", filter_name="coast")
    street = ImageCollection(base_path=r"baseDeDonneesImages", filter_name="street")
    forest = ImageCollection(base_path=r"baseDeDonneesImages", filter_name="forest")

    if view_hist:
        im_list_coast = random_image_selector(6, coast)
        images_display(im_list_coast, coast)
        histogrammes(im_list_coast, coast)

        im_list_forest = random_image_selector(6, forest)
        images_display(im_list_forest, forest)
        histogrammes(im_list_forest, forest)

        im_list_street = random_image_selector(6, street)
        images_display(im_list_street, street)
        histogrammes(im_list_street, street)

    categorized_collection = CategorizedImageCollection(coast, forest, street)

    param_1d(categorized_collection, extract_peak_hsv, dimension=0)
    param_1d(categorized_collection, extract_peak_std_hsv, dimension=0)
    param_1d(categorized_collection, extract_peak_cmyk, dimension=2)
    param_1d(categorized_collection, extract_peak_lab, subset_start=100, subset_end=150, dimension=1)
    param_1d(categorized_collection, extract_peak_lab, subset_start=100, subset_end=150, dimension=2)
    param_1d(categorized_collection, extract_peak_height_cmyk, subset_start=0, subset_end=50, dimension=0)
    param_1d(categorized_collection, extract_peak_height_cmyk, subset_start=0, subset_end=50, dimension=1)
    param_1d(categorized_collection, extract_mean_hsv, dimension=1)
    # param_3d(categorized_collection, extract_mean_count_pixel_in_slice, subset_start=0, subset_end=75, num_images=-1, title="RGB Slice sum 0 - 75")
    # param_3d(categorized_collection, extract_mean_count_pixel_in_slice, subset_start=75, subset_end=250, num_images=-1, title="RGB Slice sum 75 - 250")
    # get_images(categorized_collection, extract_param_pixels, num_images=-1, title="HSV params pixel")

    plt.show()

if __name__ == '__main__':
    main()

