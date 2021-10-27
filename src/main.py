'''
Script for image visualization

'''
import matplotlib.pyplot as plt
import numpy as np
from skimage import color as skic

from src.color_transformation import rgb_to_cmyk
from src.image_collection import ImageCollection
from src.visualization.plot_utils import plot_1d, plot_3d
from src.visualization.view_histogram import histogrammes

def main():
    forest = ImageCollection("forest")
    im_list_forest = random_image_selector(6, forest)
    images_display(im_list_forest, forest)
    histogrammes(im_list_forest, forest)

    street = ImageCollection("street")
    im_list_street = random_image_selector(6, street)
    images_display(im_list_street, street)
    histogrammes(im_list_street, street)

    coast = ImageCollection("coast")
    im_list_coast = random_image_selector(6, coast)
    images_display(im_list_coast, coast)
    histogrammes(im_list_coast, coast)

    def extract_mean(rgb):
        image = rgb_to_cmyk(rgb)
        means = [np.mean(image[:, :, 0]), np.mean(image[:, :, 1]), np.mean(image[:, :, 2])]
        return means

    def extract_std(rgb):
        image = skic.rgb2lab(rgb)
        stds = [np.std(image[:, :, 0]), np.std(image[:, :, 1]), np.std(image[:, :, 2])]
        return stds

    def extract_var(rgb):
        image = rgb
        variances = [np.var(image[:, :, 0]), np.var(image[:, :, 1]), np.var(image[:, :, 2])]
        return variances

    def extract_diff_mean(rgb):
        image = rgb
        n_bins = 256
        pixel_values = np.zeros((3, n_bins))
        for j in range(4):
            for i in range(n_bins):
                if j < 3:
                    pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)

        diff_mean = [np.max(np.diff(smooth(pixel_values[0, :], 5))), np.max(np.diff(smooth(pixel_values[1, :], 5))), np.max(np.diff(smooth(pixel_values[2, :], 5)))]
        return diff_mean

    def extract_cov(rgb):
        image = skic.rgb2lab(rgb)
        stds = [np.cov(image[:, :, 0], image[:, :, 1])[0][1], np.cov(image[:, :, 1], image[:, :, 2])[0][1], np.cov(image[:, :, 2], image[:, :, 0])[0][1]]
        return stds

    def extract_mean_ba(rgb):
        imageLab = skic.rgb2lab(rgb)
        b = np.mean(imageLab[:, :, 1])
        a = np.mean(imageLab[:, :, 2])
        return a - b


    forest_mean = extract_params(200, forest, extract_diff_mean)
    street_mean = extract_params(200, street, extract_diff_mean)
    coast_mean = extract_params(200, coast, extract_diff_mean)

    plot_3d(coast_mean, forest_mean, street_mean)


    # Extract the mean of the blue color and plot the histogram of them
    def extract_mean_saturation(rgb):
        imagehsv = skic.rgb2hsv(rgb)
        return float(np.mean(imagehsv[:, :, 1]))

    forest_mean = extract_param(200, forest, extract_mean_saturation)
    street_mean = extract_param(200, street, extract_mean_saturation)
    coast_mean = extract_param(200, coast, extract_mean_saturation)

    plot_1d(coast_mean, forest_mean, street_mean)


    # Correlation between the values of blues and red
    def extract_rb_correlation(rgb):
        hist_r, bin_r = np.histogram(rgb[:, :, 0], 255)
        hist_g, bin_g = np.histogram(rgb[:, :, 1], 255)
        cross_correlation = np.corrcoef(hist_g, hist_r)
        return float(cross_correlation[0, 1])

    forest_mean = extract_param(200, forest, extract_rb_correlation)
    street_mean = extract_param(200, street, extract_rb_correlation)
    coast_mean = extract_param(200, coast, extract_rb_correlation)

    plot_1d(coast_mean, forest_mean, street_mean)



    plt.show()

if __name__ == '__main__':
    main()

