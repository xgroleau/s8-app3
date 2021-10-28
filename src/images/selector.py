import numpy as np


def random_image_selector(number, im_coll):
    '''
    Génère une liste d'indexes pour choisir des images au hasard dans la liste
    image_list: liste de strings de 980 items
    number: int
    '''
    if number == -1:
        return np.arange(len(im_coll.image_list))

    indexes_list = np.random.randint(low=0, high=np.size(im_coll.image_list, 0), size=number)

    # Protection contre les doublons
    unique_indexes = np.unique(indexes_list)
    while np.size(unique_indexes) != number:
        extra_indexes = np.random.randint(low=0, high=np.size(im_coll.image_list, 0), size=number - np.size(unique_indexes))
        new_array = np.append(unique_indexes, extra_indexes)
        unique_indexes = np.unique(new_array)

    return np.sort(unique_indexes)