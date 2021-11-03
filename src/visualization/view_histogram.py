import matplotlib.pyplot as plt
from skimage import color as skic
import numpy as np
from PIL import Image
from src.images.color_transformation import rgb_to_cmyk, smooth
from src.images.image_collection import ImageCollection


def histogrammes(indexes: iter, im_coll: ImageCollection, n_bins: int = 256):
    '''
    Takes images array and an index to pick the
    appropriate image
    images format: (number of image, Width, Height, RGB channel)
    index: int or list of int
    '''
    if type(indexes) == int:
        indexes = [indexes]

    fig = plt.figure()
    ax = fig.subplots(len(indexes),4)

    for num_images in range(len(indexes)):

        imageRGB = np.array(Image.open(im_coll.image_folder + '\\' + im_coll.image_list[indexes[num_images]]))
        imageLab = skic.rgb2lab(imageRGB)
        imageHSV = skic.rgb2hsv(imageRGB)
        imageCMYK = rgb_to_cmyk(imageRGB)


        #Lab
        min_L = 0
        max_L = 100
        min_ab = -110
        max_ab = 110
        imageLabhist = np.zeros(imageLab.shape)
        imageLabhist[:,:,0] = np.round(imageLab[:,:,0]*(n_bins-1)/max_L) #L has all values between 0 and 100 skic.rgb2lab
        imageLabhist[:,:,1] = np.round((imageLab[:,:,1]-min_ab)*(n_bins-1)/(max_ab-min_ab)) #ab has all values between -110 and 110
        imageLabhist[:, :, 2] = np.round((imageLab[:, :, 2] - min_ab) * (n_bins - 1) / (max_ab - min_ab))  # ab has all values between -110 and 110

        imageHSVhist = np.round(imageHSV*(n_bins-1)) #HSV has all values between 0 and 100

        # A list per color channel
        pixel_valuesRGB = np.zeros((3,n_bins))
        pixel_valuesLab = np.zeros((3,n_bins))
        pixel_valuesHSV = np.zeros((3,n_bins))
        pixel_valuesCMYK = np.zeros((4,n_bins))

        for i in range(n_bins):
            for j in range(4):
                if j < 3:
                    pixel_valuesRGB[j,i] = np.count_nonzero(imageRGB[:,:,j]==i)
                    pixel_valuesLab[j,i] = np.count_nonzero(imageLabhist[:,:,j]==i)
                    pixel_valuesHSV[j,i] = np.count_nonzero(imageHSVhist[:,:,j]==i)
                pixel_valuesCMYK[j,i] = np.count_nonzero(imageCMYK[:,:,j]==i)

        skip = 5
        start = skip
        end = n_bins-skip

        ax[num_images,0].scatter(range(start,end), pixel_valuesRGB[0,start:end], c='red')
        ax[num_images,0].scatter(range(start,end), pixel_valuesRGB[1,start:end], c='green')
        ax[num_images,0].scatter(range(start,end), pixel_valuesRGB[2,start:end], c='blue')
        ax[num_images,0].set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
        # ajouter le titre de la photo observée dans le titre de l'histograme
        image_name = im_coll.image_list[indexes[num_images]]
        ax[num_images,0].set_title(f'histogramme RGB de {image_name}')

        ax[num_images,1].scatter(range(start,end), pixel_valuesHSV[0,start:end], c='orange')
        ax[num_images,1].scatter(range(start,end), pixel_valuesHSV[1,start:end], c='cyan')
        ax[num_images,1].scatter(range(start,end), pixel_valuesHSV[2,start:end], c='magenta')
        ax[num_images,1].set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
        ax[num_images,1].set_title(f'histogramme HSV de {image_name}')

        x_values = range(start, end)
        ax[num_images, 2].plot(x_values, smooth(pixel_valuesRGB[0, start:end], 5), c='red')
        ax[num_images, 2].plot(x_values, smooth(pixel_valuesRGB[1, start:end], 5), c='green')
        ax[num_images, 2].plot(x_values, smooth(pixel_valuesRGB[2, start:end], 5), c='blue')
        ax[num_images, 2].set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
        ax[num_images, 2].set_title(f'histogramme LAB de {image_name}')

        ax[num_images, 3].scatter(range(start, end), pixel_valuesCMYK[0, start:end], c='cyan')
        ax[num_images, 3].scatter(range(start, end), pixel_valuesCMYK[1, start:end], c='magenta')
        ax[num_images, 3].scatter(range(start, end), pixel_valuesCMYK[2, start:end], c='yellow')
        ax[num_images, 3].scatter(range(start, end), pixel_valuesCMYK[3, start:end], c='grey')
        ax[num_images, 3].set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
        ax[num_images, 3].set_title(f'histogramme CMYK de {image_name}')

    return None
