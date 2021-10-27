import matplotlib.pyplot as plt
from PIL import Image


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
