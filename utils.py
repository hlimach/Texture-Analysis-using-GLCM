import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from fast_glcm import *

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (255,0,0)
FEATURES = {1: 'mean', 
            2: 'std', 
            3: 'contrast', 
            4: 'dissimilarity', 
            5: 'homogeneity', 
            6: 'asm',
            7: 'energy',
            8: 'max_',
            9: 'entropy'}

def load_pngs_dict(path):
    '''
    use path and create dict of id and images
    '''
    imgs_ids = {}
    
    for file in os.listdir(path):
        if file.endswith('.png'):
            img_file = os.path.join(path, file)
            id = file.split('.')[0]

            img = cv2.imread(img_file)
            imgs_ids[id] = img

    return imgs_ids

def get_crop_image(img, h, w):
    '''
    crop img to fit tiling requirement
    '''
    height, width = img.shape[:2]

    new_width = width - (width % w)
    new_height = height - (height % h)

    img = img[0:new_height, 0:new_width]

    return img

def get_tiles_by_grid(img, r=5, c=10):
    '''
    Parameters
    ----------
    img: shape=(h,w,ch), dtype=np.uint8
        input image
    r: int
        number of rows in grid
    c: int
        number of columns in grid

    Returns
    -------
    Cropped image, shape = (h,w,ch)
    Dict of tiles generated, {id: tile}
    Image masked with tiles and ids, shape = (h,w,ch)
    '''

    img = get_crop_image(img, r, c)
    height, width = img.shape[:2]

    tile_w = int(width/c)
    tile_h = int(height/r)

    mask = img.copy()
    id = 0
    tiles = {}

    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            cv2.rectangle(mask, pt1=(x,y), pt2=(x+tile_w-1,y+tile_h-1), color=COLOR, thickness=1)
            cv2.putText(mask, str(id), (x,y+tile_h), FONT, 1, COLOR, 3, cv2.LINE_AA)
            tile = img[y:y+tile_h, x:x+tile_w]
            tiles[id] = tile
            id = id + 1

    return img, tiles, mask

def get_tiles_by_pixels(img, h=100, w=100):
    '''
    Parameters
    ----------
    img: shape=(h,w,ch), dtype=np.uint8
        input image
    h: int
        height of tile in pixels
    w: int
        width of tile in pixels

    Returns
    -------
    Cropped image, shape = (h,w,ch)
    Dict of tiles generated, {id: tile}
    Image masked with tiles and ids, shape = (h,w,ch)
    '''

    img = get_crop_image(img, h, w)
    height, width = img.shape[:2]
    mask = img.copy()

    id = 0
    tiles = {}

    for y in range(0, height, h):
        for x in range(0, width, w):
            cv2.rectangle(mask, pt1=(x,y), pt2=(x+w-1,y+h-1), color=COLOR, thickness=1)
            cv2.putText(mask, str(id), (x,y+h), FONT, 1, COLOR, 3, cv2.LINE_AA)
            tile = img[y:y+h, x:x+w]
            tiles[id] = tile
            id = id + 1

    return img, tiles, mask

def calc_glcm_features(img):
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

def display_glcm_features(glcm_features, id):
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(id)

    for key in FEATURES:
        ax = plt.subplot(3,3,key)
        ax.set_title(FEATURES[key])
        plt.imshow(glcm_features[FEATURES[key]], cmap = 'gray')
    plt.show()

def get_avg_features(glcm_features):
    avg_features = {}
    for key in glcm_features:
        avg_features[key] = np.average(glcm_features[key])
    return avg_features

def calc_avg_tile_features(tiles):
    avg_tile_features = {}
    for id in tiles:
        glcm_features = calc_glcm_features(tiles[id])
        avg_tile_features[id] = get_avg_features(glcm_features)
    return avg_tile_features

def print_feature_graph(avg_tile_features):
    fig = plt.figure(figsize=(22, 18))

    for key in FEATURES:
        ax = plt.subplot(3,3,key)
        ax.set_title(FEATURES[key])
        x = [x for x in avg_tile_features]
        y = [avg_tile_features[x][FEATURES[key]] for x in avg_tile_features]
        plt.scatter(x, y)
    plt.show()