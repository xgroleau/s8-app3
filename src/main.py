'''
Script for image visualization

'''
import matplotlib.pyplot as plt

from src.images import random_image_selector
from src.images.image_collection import ImageCollection, CategorizedImageCollection
from src.params.extract_param import extract_mean_saturation, extract_rb_correlation, extract_var
from src.params.param import param_1d, param_3d
from src.visualization import images_display
from src.visualization.view_histogram import histogrammes


def main():
    view_hist = True
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

    #param_1d(categorized_collection, extract_mean_saturation)
    #param_1d(categorized_collection, extract_rb_correlation)
    param_3d(categorized_collection, extract_var, "Variance")

    plt.show()

if __name__ == '__main__':
    main()

