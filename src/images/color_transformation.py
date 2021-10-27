from PIL import Image
import numpy as np

def rgb_to_cmyk(rgb_img):
    # Make float and divide by 255 to give BGRdash
    rgbdash = rgb_img.astype(np.float) / 255.
    epsilon = 1e-12

    # Calculate K as (1 - whatever is biggest out of Rdash, Gdash, Bdash)
    K = 1 - np.max(rgbdash, axis=2) - epsilon

    # Calculate C
    C = (1 - rgbdash[..., 0] - K) / (1 - K)

    # Calculate M
    M = (1 - rgbdash[..., 1] - K) / (1 - K)

    # Calculate Y
    Y = (1 - rgbdash[..., 2] - K) / (1 - K)

    # Combine 4 channels into single image and re-scale back up to uint8
    return (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)