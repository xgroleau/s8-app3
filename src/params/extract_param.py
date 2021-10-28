import numpy as np
from skimage import color as skic

from src.images import rgb_to_cmyk, smooth


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

    diff_mean = [np.max(np.diff(smooth(pixel_values[0, :], 5))), np.max(np.diff(smooth(pixel_values[1, :], 5))),
                 np.max(np.diff(smooth(pixel_values[2, :], 5)))]
    return diff_mean


def extract_cov(rgb):
    image = skic.rgb2lab(rgb)
    stds = [np.cov(image[:, :, 0], image[:, :, 1])[0][1], np.cov(image[:, :, 1], image[:, :, 2])[0][1],
            np.cov(image[:, :, 2], image[:, :, 0])[0][1]]
    return stds


# Extract the mean of b minus a in LAB representation
def extract_mean_ba(rgb):
    imageLab = skic.rgb2lab(rgb)
    b = np.mean(imageLab[:, :, 1])
    a = np.mean(imageLab[:, :, 2])
    return a - b


# Extract the mean of the saturation in an image
def extract_mean_saturation(rgb):
    imagehsv = skic.rgb2hsv(rgb)
    return float(np.mean(imagehsv[:, :, 1]))


# Extract the mean of the hue in an image
def extract_mean_hue(rgb):
    imagehsv = skic.rgb2hsv(rgb)
    return float(np.mean(imagehsv[:, :, 0]))


# Correlation between the values of blues and red (in RGB)
def extract_rb_correlation(rgb):
    hist_r, bin_r = np.histogram(rgb[:, :, 0], 255)
    hist_g, bin_g = np.histogram(rgb[:, :, 1], 255)
    cross_correlation = np.corrcoef(hist_g, hist_r)
    return float(cross_correlation[0, 1])


# Extract the peak of b minus a in LAB representation
def extract_peak_b_minus_a(rgb):
    imageLab = skic.rgb2lab(rgb)
    b_hist, b_bin = np.histogram(imageLab[:, :, 1], 255)
    a_hist, a_bin = np.histogram(imageLab[:, :, 2], 255)
    b_hist_sliced = b_hist[100:200]
    a_hist_sliced = a_hist[100:200]
    return np.mean(b_hist_sliced - a_hist_sliced)
