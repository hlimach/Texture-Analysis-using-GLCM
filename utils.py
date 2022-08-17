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

def load_pngs_dict(img_path, mask_path):
    '''
    create dict of id with image and its pothole mask
    '''
    imgs_ids = {}
    
    for file in os.listdir(img_path):
        if file.endswith('.png'):
            id = file.split('.')[0]

            img = cv2.imread(os.path.join(img_path, file))
            mask = cv2.imread(os.path.join(mask_path, id+'_mask.png'))
            imgs_ids[id] = (img, mask)

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

def get_labelled_tiles(img, mask, h=100, w=100):
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
    Dict of tiles generated, {id: tile img}
    Dict of labels, {tile id, 'label'}
    Image masked with tiles, labels and ids, shape = (h,w,ch)
    '''

    img = get_crop_image(img, h, w)
    mask = get_crop_image(mask, h, w)
    grid = img.copy()

    height, width = img.shape[:2]

    id = 0
    tiles = {}
    labels = {}
    threshold = int(h*w*0.6)

    for y in range(0, height, h):
        for x in range(0, width, w):
            m_tile = mask[y:y+h, x:x+w]
            i_tile = img[y:y+h, x:x+w]

            if np.count_nonzero(m_tile==0) >= threshold:
                tiles[id] = i_tile
                labels[id] = 'Pavement'
            else:
                tiles[id] = i_tile
                labels[id] = 'Pothole'

                # add a transparent rectangle on grid to indicate label
                g_tile = grid[y:y+h, x:x+w]
                white_rect = np.ones(g_tile.shape, dtype=np.uint8) * 255
                res = cv2.addWeighted(g_tile, 0.5, white_rect, 0.5, 1.0)
                grid[y:y+h, x:x+w] = res
            
            id = id + 1
            cv2.rectangle(grid, pt1=(x,y), pt2=(x+w-1,y+h-1), color=COLOR, thickness=1)
            cv2.putText(grid, str(id), (x,y+h), FONT, 1, COLOR, 1, cv2.LINE_AA)

    return img, tiles, labels, grid

def calc_glcm_features(img):
    # returns {'name of feature', value}
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
    # returns {'name of feature', avg value}
    avg_features = {}
    for key in glcm_features:
        avg_features[key] = np.average(glcm_features[key])
    return avg_features

def calc_avg_tile_features(tiles):
    # for each tile, calc the glcm features and then get avg values
    # returns tile id, dict of {'name of feature', avg value}
    avg_tile_features = {}
    for id in tiles:
        glcm_features = calc_glcm_features(tiles[id])
        avg_tile_features[id] = get_avg_features(glcm_features)
    return avg_tile_features

def visualize_labelled_data(avg_tile_features, tile_labels):
    fig = plt.figure(figsize=(22, 18))
    colors = ('black', 'red')
    labels = ('Pavement', 'Pothole')

    for key in FEATURES:
        ax = fig.add_subplot(3,3,key)
        ax.set_title(FEATURES[key])

        for i in range(len(labels)):
            x = [id for id in tile_labels if tile_labels[id]==labels[i]]
            y = [avg_tile_features[id][FEATURES[key]] for id in x]
            ax.scatter(x, y, c=colors[i], edgecolors='none', s=30, label=labels[i])
        plt.legend()
    plt.show()

def display_histogram(avg_tile_features, tile_labels, bin_count=10):
    fig = plt.figure(figsize=(22, 18))
    colors = ('black', 'red')
    labels = ('Pavement', 'Pothole')

    for key in FEATURES:
        ax = fig.add_subplot(3,3,key)
        ax.set_title(FEATURES[key])

        y0 = [avg_tile_features[id][FEATURES[key]] for id in tile_labels if tile_labels[id]==labels[0]]
        y1 = [avg_tile_features[id][FEATURES[key]] for id in tile_labels if tile_labels[id]==labels[1]]

        plt.hist([y0,y1], bin_count, color=colors, label=labels)
        plt.legend()
    plt.show()