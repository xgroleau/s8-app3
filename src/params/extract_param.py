import numpy as np
from skimage import color as skic

from src.images import rgb_to_cmyk, smooth


"""
Definition of the function to extract parameters
Those functions takes an RGB image and returns the extracted parameters of the image
"""


def extract_mean(rgb):
    image = rgb_to_cmyk(rgb)
    n_bins = 256
    # image = np.round(imageHSV*(n_bins-1)) #HSV has all values between 0 and 100
    means = [np.mean(image[:, :, 0]), np.mean(image[:, :, 1]), np.mean(image[:, :, 3])]
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
    image = rgb_to_cmyk(rgb)
    stds = [np.cov(image[:, :, 0], image[:, :, 3])[0][1], np.cov(image[:, :, 3], image[:, :, 2])[0][1],
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


def get_range_around_index(index_max, width=4):
    index = index_max
    if index < width:
        index = width
    elif index > 255 - width:
        index = 255 - width
    start = index - width
    end = index + width
    return start, end


def extract_peak_color(rgb, subset_start=50, subset_end=100):
    image = rgb
    n_bins = 256

    # min_L = 0
    # max_L = 100
    # min_ab = -110
    # max_ab = 110
    # image = np.zeros(imageLab.shape)
    # image[:, :, 0] = np.round(
    #     imageLab[:, :, 0] * (n_bins - 1) / max_L)  # L has all values between 0 and 100 skic.rgb2lab
    # image[:, :, 1] = np.round(
    #     (imageLab[:, :, 1] - min_ab) * (n_bins - 1) / (max_ab - min_ab))  # ab has all values between -110 and 110
    # image[:, :, 2] = np.round(
    #     (imageLab[:, :, 2] - min_ab) * (n_bins - 1) / (max_ab - min_ab))  # ab has all values between -110 and 110
    # image = np.round(imageHSV*(n_bins-1)) #HSV has all values between 0 and 100
    pixel_values = np.zeros((4, n_bins))
    for j in range(3):
        for i in range(n_bins):
            pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)

    index_max_x = np.argmax(smooth(pixel_values[0, :], 5)[subset_start:subset_end]) + subset_start
    index_max_y = np.argmax(smooth(pixel_values[1, :], 5)[subset_start:subset_end]) + subset_start
    index_max_z = np.argmax(smooth(pixel_values[2, :], 5)[subset_start:subset_end]) + subset_start

    peaks = [smooth(pixel_values[0, :], 5)[index_max_x],
             smooth(pixel_values[1, :], 5)[index_max_y],
             smooth(pixel_values[2, :], 5)[index_max_z]]
    return peaks


def extract_mean_count_pixel_in_slice(rgb, subset_start=0, subset_end=256, dimension=0):
    image = rgb
    n_bins = 256
    # image = np.round(imageHSV*(n_bins-1)) #HSV has all values between 0 and 100

    pixel_values = np.zeros((4, n_bins))

    for j in range(3):
        for i in range(n_bins):
            pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)

    return np.sum(pixel_values[dimension, subset_start:subset_end])


def extract_param_pixels(rgb, dimension=0):
    imageHSV = skic.rgb2hsv(rgb)
    n_bins = 256
    image = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100 skic.rgb2hsv

    # min_L = 0
    # max_L = 100
    # min_ab = -110
    # max_ab = 110
    # image = np.zeros(imageLab.shape)
    # image[:, :, 0] = np.round(
    #     imageLab[:, :, 0] * (n_bins - 1) / max_L)  # L has all values between 0 and 100 skic.rgb2lab
    # image[:, :, 1] = np.round(
    #     (imageLab[:, :, 1] - min_ab) * (n_bins - 1) / (max_ab - min_ab))  # ab has all values between -110 and 110
    # image[:, :, 2] = np.round(
    #     (imageLab[:, :, 2] - min_ab) * (n_bins - 1) / (max_ab - min_ab))  # ab has all values between -110 and 110
    pixel_values = np.zeros((4, n_bins))

    for j in range(3):
        for i in range(n_bins):
            pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)

    subset_start = 0
    subset_end = 256
    index_max_x = np.argmax(smooth(pixel_values[dimension, :], 5)[subset_start:subset_end]) + subset_start
    sx, ex = get_range_around_index(index_max_x)

    means = [index_max_x,
             pixel_values[dimension, sx:ex].mean(),
             pixel_values[dimension, sx:ex].std()]

    return means


def extract_cov_pixels(rgb, dimension=0):
    imageHSV = skic.rgb2hsv(rgb)
    n_bins = 256
    image = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100 skic.rgb2hsv

    # min_L = 0
    # max_L = 100
    # min_ab = -110
    # max_ab = 110
    # image = np.zeros(imageLab.shape)
    # image[:, :, 0] = np.round(
    #     imageLab[:, :, 0] * (n_bins - 1) / max_L)  # L has all values between 0 and 100 skic.rgb2lab
    # image[:, :, 1] = np.round(
    #     (imageLab[:, :, 1] - min_ab) * (n_bins - 1) / (max_ab - min_ab))  # ab has all values between -110 and 110
    # image[:, :, 2] = np.round(
    #     (imageLab[:, :, 2] - min_ab) * (n_bins - 1) / (max_ab - min_ab))  # ab has all values between -110 and 110
    pixel_values = np.zeros((4, n_bins))

    for j in range(3):
        for i in range(n_bins):
            pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)

    subset_start = 5
    subset_end = 251
    index_max_x = np.argmax(smooth(pixel_values[dimension, :], 5)[subset_start:subset_end]) + subset_start
    sx, ex = get_range_around_index(index_max_x, 10)

    means = [np.cov(pixel_values[0, sx:ex], pixel_values[1, sx:ex])[0, 1],
             np.cov(pixel_values[1, sx:ex], pixel_values[2, sx:ex])[0, 1],
             np.cov(pixel_values[0, sx:ex], pixel_values[2, sx:ex])[0, 1]]

    return means


def extract_mean_hsv(rgb, dimension=0, subset_start=0, subset_end=256):
    image = skic.rgb2hsv(rgb)
    n_bins = 256
    # image = np.round(imageHSV*(n_bins-1)) #HSV has all values between 0 and 100
    means = np.mean(image[:, :, dimension])
    return means


def extract_mean_cmyk(rgb, dimension=0, subset_start=0, subset_end=256):
    image = rgb_to_cmyk(rgb)
    means = np.mean(image[:, subset_start:subset_end, dimension])
    return means


def extract_mean_rgb(rgb, subset_start=0, subset_end=256, dimension=0):
    means = np.mean(rgb[:, subset_start:subset_end, dimension])
    return means


def extract_peak_height_cmyk(rgb, subset_start=0, subset_end=256, dimension=0):
    image = rgb_to_cmyk(rgb)
    n_bins = 256

    pixel_values = np.zeros((4, n_bins))
    for j in range(4):
        for i in range(n_bins):
            pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)

    index_max_x = np.argmax(smooth(pixel_values[dimension, :], 5)[subset_start:subset_end]) + subset_start

    peak = smooth(pixel_values[dimension, :], 5)[index_max_x]

    return peak


def extract_peak_cmyk(rgb, subset_start=0, subset_end=256, dimension=0):
    image = rgb_to_cmyk(rgb)
    n_bins = 256

    pixel_values = np.zeros((4, n_bins))
    for j in range(4):
        for i in range(n_bins):
            pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)

    index_max_x = np.argmax(smooth(pixel_values[dimension, :], 5)[subset_start:subset_end]) + subset_start

    return index_max_x


def extract_peak_lab(rgb, subset_start=0, subset_end=256, dimension=0):
    imageLab = skic.rgb2lab(rgb)
    n_bins = 256
    min_L = 0
    max_L = 100
    min_ab = -110
    max_ab = 110
    image = np.zeros(imageLab.shape)
    image[:, :, 0] = np.round(
        imageLab[:, :, 0] * (n_bins - 1) / max_L)  # L has all values between 0 and 100 skic.rgb2lab
    image[:, :, 1] = np.round(
        (imageLab[:, :, 1] - min_ab) * (n_bins - 1) / (max_ab - min_ab))  # ab has all values between -110 and 110
    image[:, :, 2] = np.round(
        (imageLab[:, :, 2] - min_ab) * (n_bins - 1) / (max_ab - min_ab))  # ab has all values between -110 and 110

    pixel_values = np.zeros((3, n_bins))
    for j in range(3):
        for i in range(n_bins):
            pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)

    index_max_x = np.argmax(smooth(pixel_values[dimension, :], 5)[subset_start:subset_end]) + subset_start

    return index_max_x


def extract_peak_hsv(rgb, subset_start=0, subset_end=256, dimension=0):
    imageHSV = skic.rgb2hsv(rgb)
    n_bins = 256
    image = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100 skic.rgb2hsv

    pixel_values = np.zeros((3, n_bins))
    for j in range(3):
        for i in range(n_bins):
            pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)

    index_max_x = np.argmax(smooth(pixel_values[dimension, :], 5)[subset_start:subset_end]) + subset_start

    return index_max_x


def extract_peak_std_hsv(rgb, subset_start=0, subset_end=256, dimension=0):
    imageHSV = skic.rgb2hsv(rgb)
    n_bins = 256
    image = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100 skic.rgb2hsv

    pixel_values = np.zeros((3, n_bins))
    for j in range(3):
        for i in range(n_bins):
            pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)

    index_max_x = np.argmax(smooth(pixel_values[dimension, :], 5)[subset_start:subset_end]) + subset_start

    sx, ex = get_range_around_index(index_max_x)

    std = pixel_values[dimension, sx:ex].std()
    return std


def extract_light_pixel_count(rgb):
    value_3 = 0
    for i in range(len(rgb)):
        for j in range(len(rgb[i])):
            if rgb[i, j, 0] > 200 and rgb[i, j, 1] > 200 and rgb[i, j, 2] > 200:
                value_3 += 1

    return value_3


def extractor_cmyk(rgb, dimension=0):
    image = rgb_to_cmyk(rgb)

    image.reshape(-1, 4)
    unique, counts = np.unique(image.reshape(-1, 4), axis=0, return_counts=True)
    value = [0, 0, 0, 0]
    value[0], value[1], value[2], value[3] = unique[np.argmax(counts)]
    return value[dimension]


def rgb_value(rgb):
    return rgb


def extractor_mean(rgb, dimension=0, base_function=rgb_value):
    image = base_function(rgb)
    return np.mean(image[:, :, dimension])


def extractor_std(rgb, dimension=0, base_function=rgb_value):
    image = base_function(rgb)
    return np.std(image[:, :, dimension])


def extractor_median(rgb, dimension=0, base_function=rgb_value):
    image = base_function(rgb)
    return np.median(image[:, :, dimension])


def extractor_unique(rgb, dimension=0, base_function=rgb_value):
    image = base_function(rgb)
    dimensions = image.shape[2]
    image.reshape(-1, dimensions)
    unique, counts = np.unique(image.reshape(-1, dimensions), axis=0, return_counts=True)
    value = unique[np.argmax(counts)]
    return value[dimension]