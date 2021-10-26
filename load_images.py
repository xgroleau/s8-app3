'''
Script for image visualization

'''

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from PIL import Image

# liste de toutes les images
path = glob.glob(r".\baseDeDonneesImages\*.jpg")
image_folder = r".\baseDeDonneesImages"
image_list = os.listdir(image_folder)
# Filtrer pour juste garder les images
image_list = [i for i in image_list if '.jpg' in i]

# Créer un array qui contient toutes les images
# Dimensions [980, 256, 256, 3]
# Valeurs    [# image, hauteur, largeur, RGB]
#images = np.array([np.array(Image.open(image)) for image in path])


def histogrammes(image_list, indexes=1):
    '''
    Takes images array and an index to pick the
    appropriate image
    images format: (number of image, Width, Height, RGB channel)
    index: int or list of int
    '''
    if type(indexes) == int:
        indexes = [indexes]
        
    fig = plt.figure()
    ax = fig.subplots(len(indexes))


    for num_images in range(len(indexes)):
    
        image = np.array(Image.open(image_folder + '\\' + image_list[indexes[num_images]]))

        # Number of bins per color
        n_bins = 256
    
        # A list per color channel
        pixel_values = np.zeros((3,n_bins))
    
        for i in range(n_bins):
            pixel_values[0,i] = np.count_nonzero(image[:,:,0]==i)
            pixel_values[1,i] = np.count_nonzero(image[:,:,1]==i)
            pixel_values[2,i] = np.count_nonzero(image[:,:,2]==i)

        #print('pixel values: ', pixel_values[0,:])
        #print('sum pixels: ', np.sum(pixel_values[0,:]))
        ax[num_images].scatter(range(n_bins), pixel_values[0,:], c='red')
        ax[num_images].scatter(range(n_bins), pixel_values[1,:], c='green')
        ax[num_images].scatter(range(n_bins), pixel_values[2,:], c='blue')
        ax[num_images].set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
        # ajouter le titre de la photo observée dans le titre de l'histograme
        image_name = image_list[indexes[num_images]]
        ax[num_images].set_title(f'histogramme de {image_name}')
    plt.show()

    return None


def random_image_selector(image_list, number):
    '''
    Génère une liste d'indexes pour choisir des images au hasard dans la liste
    image_list: liste de strings de 980 items
    number: int
    '''
    indexes_list = np.random.randint(low=0, high=np.size(image_list, 0), size=number)

    # Protection contre les doublons
    unique_indexes = np.unique(indexes_list)
    while np.size(unique_indexes) != number:
        extra_indexes = np.random.randint(low=0, high=np.size(image_list, 0), size=number - np.size(unique_indexes))
        new_array = np.append(unique_indexes, extra_indexes)
        unique_indexes = np.unique(new_array)

    return np.sort(unique_indexes)


def images_display(image_list, indexes=1):
    '''
    fonction pour afficher les images correspondant aux indices

    :param image_list: liste d'image à montrer
    :param indexes: indices de la liste d'image
    :return: None
    '''
    if type(indexes) == int:
        indexes = [indexes]

    for index in indexes:
        im = Image.open(image_folder + '\\' + image_list[index])
        im.show()

# ============= Script principal ======================== #

# Appeler ici les fonctions


im_list = random_image_selector(image_list, 6)
print(im_list)
#images_display(image_list, range(6))
#images_display(image_list, im_list)

#histogrammes(image_list, range(6))
histogrammes(image_list, im_list)
