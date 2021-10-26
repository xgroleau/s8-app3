from PIL import ImageCms


#https://stackoverflow.com/questions/13405956/convert-an-image-rgb-lab-with-python

def rgb_to_lab(rgb_img):
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p = ImageCms.createProfile("LAB")
    rgb2lab_trans = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    return ImageCms.applyTransform(rgb_img, rgb2lab_trans)