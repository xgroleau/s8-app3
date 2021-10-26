'''
Script for image visualization

'''
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from PIL import Image
from skimage import color as skic

from color_transformation import rgb_to_cmyk
from plot_utils import plot_1d


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

        # Créer un array qui contient toutes les images
        # Dimensions [980, 256, 256, 3]
        # Valeurs    [# image, hauteur, largeur, RGB]
        #images = np.array([np.array(Image.open(image)) for image in path])

def histogrammes(indexes, im_coll):
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
      
        # Number of bins per color channel
        n_bins = 256

        #Lab
        min_L = 0
        max_L = 100
        min_ab = -110
        max_ab = 110
        imageLabhist = np.zeros(imageLab.shape)
        imageLabhist[:,:,0] = np.round(imageLab[:,:,0]*(n_bins-1)/max_L) #L has all values between 0 and 100
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
        #print('pixel values: ', pixel_values[0,:])
        #print('sum pixels: ', np.sum(pixel_values[0,:]))
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

        ax[num_images, 2].scatter(range(start, end), pixel_valuesLab[0, start:end], c='grey')
        ax[num_images, 2].scatter(range(start, end), pixel_valuesLab[1, start:end], c='orange')
        ax[num_images, 2].scatter(range(start, end), pixel_valuesLab[2, start:end], c='violet')
        ax[num_images, 2].set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
        ax[num_images, 2].set_title(f'histogramme LAB de {image_name}')

        ax[num_images, 3].scatter(range(start, end), pixel_valuesCMYK[0, start:end], c='cyan')
        ax[num_images, 3].scatter(range(start, end), pixel_valuesCMYK[1, start:end], c='magenta')
        ax[num_images, 3].scatter(range(start, end), pixel_valuesCMYK[2, start:end], c='yellow')
        ax[num_images, 3].scatter(range(start, end), pixel_valuesCMYK[3, start:end], c='grey')
        ax[num_images, 3].set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
        ax[num_images, 3].set_title(f'histogramme CMYK de {image_name}')

    return None


def random_image_selector(number, im_coll):
    '''
    Génère une liste d'indexes pour choisir des images au hasard dans la liste
    image_list: liste de strings de 980 items
    number: int
    '''
    indexes_list = np.random.randint(low=0, high=np.size(im_coll.image_list, 0), size=number)

    # Protection contre les doublons
    unique_indexes = np.unique(indexes_list)
    while np.size(unique_indexes) != number:
        extra_indexes = np.random.randint(low=0, high=np.size(im_coll.image_list, 0), size=number - np.size(unique_indexes))
        new_array = np.append(unique_indexes, extra_indexes)
        unique_indexes = np.unique(new_array)

    return np.sort(unique_indexes)


def images_display(indexes, im_coll):
    '''
    fonction pour afficher les images correspondant aux indices

    :param image_list: liste d'image à montrer
    :param indexes: indices de la liste d'image
    :return: None
    '''
    if type(indexes) == int:
        indexes = [indexes]

    fig2 = plt.figure()
    ax2 = fig2.subplots(len(indexes),1)

    for i in range(len(indexes)):
        im = Image.open(im_coll.image_folder + '\\' + im_coll.image_list[indexes[i]])
        ax2[i].imshow(im) 


def extract_param(num: int, im_coll: ImageCollection, fun: Callable[[np.ndarray], Union[float]]):
    indexes = random_image_selector(num, im_coll)
    val = np.zeros(len(indexes))
    for i in range(len(indexes)):
        imageRGB = np.array(Image.open(im_coll.image_folder + '\\' + im_coll.image_list[indexes[i]]))
        val[i] = fun(imageRGB)
    return val

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

    def extract_mean_ba(rgb):
        imageLab = skic.rgb2lab(rgb)
        b = np.mean(imageLab[:, :, 1])
        a = np.mean(imageLab[:, :, 2])
        return a - b

    def extract_mean_saturation(rgb):
        imagehsv = skic.rgb2hsv(rgb)
        return np.mean(imagehsv[:, :, 1])

    # Extract the mean of the blue color and plot the histogram of them
    forest_mean = extract_param(200, forest, extract_mean_ba)
    street_mean = extract_param(200, street, extract_mean_ba)
    coast_mean = extract_param(200, coast, extract_mean_ba)

    plot_1d(coast_mean, forest_mean, street_mean)

    # Extract the mean of the blue color and plot the histogram of them
    forest_mean = extract_param(200, forest, extract_mean_saturation)
    street_mean = extract_param(200, street, extract_mean_saturation)
    coast_mean = extract_param(200, coast, extract_mean_saturation)

    plot_1d(coast_mean, forest_mean, street_mean)

    plt.show()

if __name__ == '__main__':
    main()

