import numpy as np
from matplotlib import pyplot as plt

from fast_glcm import *

FEATURES = {1: 'mean', 
            2: 'std', 
            3: 'contrast', 
            4: 'dissimilarity', 
            5: 'homogeneity', 
            6: 'asm',
            7: 'energy',
            8: 'max_',
            9: 'entropy'}

def calc_glcm_features(img):
    '''
    Parameters
    ----------
    img: shape=(h,w,ch), dtype=np.uint8
        input image

    Returns
    -------
    Dict containing {'name of feature', glcm output}
    '''
    img = img[:,:,0]

    mean = fast_glcm_mean(img)
    std = fast_glcm_std(img)
    contrast = fast_glcm_contrast(img)
    dissimilarity = fast_glcm_dissimilarity(img)
    homogeneity = fast_glcm_homogeneity(img)
    asm, energy = fast_glcm_ASM(img)
    max_ = fast_glcm_max(img)
    entropy = fast_glcm_entropy(img)

    glcm_features = {}
    for key in FEATURES:
        glcm_features[FEATURES[key]] = eval(FEATURES[key])

    return glcm_features

def get_avg_features(glcm_features):
    '''
    Parameters
    ----------
    glcm_features: Dict containing {'name of feature', glcm output}

    Returns
    -------
    Dict containing {'name of feature', avg value}
    '''
    avg_features = {}
    for key in glcm_features:
        avg_features[key] = np.average(glcm_features[key])
    return avg_features

def calc_avg_tile_features(tiles):
    '''
    Parameters
    ----------
    tiles: Dict containing {tile id: tile img}

    Returns
    -------
    Dict of dict containing  {tile id: avg values dict}, where
    avg values dict is {'name of feature', avg value}
    '''
    avg_tile_features = {}
    for id in tiles:
        glcm_features = calc_glcm_features(tiles[id])
        avg_tile_features[id] = get_avg_features(glcm_features)
    return avg_tile_features

def display_glcm_features(glcm_features, id):
    '''
    Parameters
    ----------
    glcm_features: Dict containing {'name of feature', glcm output}
    id: tile id associated with the feature dict

    Returns
    -------
    Displays the all GLCM features as a 3x3 grid subplot
    '''
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(id)

    for key in FEATURES:
        ax = plt.subplot(3,3,key)
        ax.set_title(FEATURES[key])
        plt.imshow(glcm_features[FEATURES[key]], cmap = 'gray')
    plt.show()