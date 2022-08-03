import os
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (255,0,0)

def load_pngs_dict(path):
    '''
    reads images from given path and stores along with their id in a dict
    returns: dict of id and image
    '''
    imgs_ids = {}
    
    for file in os.listdir(path):
        if file.endswith('.png'):
            img_file = os.path.join(path, file)
            id = file.split('.')[0]

            img = cv2.imread(img_file)
            imgs_ids[id] = img

    return imgs_ids

def get_tiles(img, n=10, m=5):
    '''
    Crops image to make height divisible by 5 and width by 10 (default)
    Generates tiles from cropped image using the n and m values
    Generates masked image with tiles printed for visual guide
    returns: cropped image, dict of tiles with their id, masked image showing tiles and ids
    '''

    height, width = img.shape[:2]

    new_width = width - (width % n)
    new_height = height - (height % m)

    img = img[0:new_height, 0:new_width]

    tile_w = int(new_width/n)
    tile_h = int(new_height/m)

    mask = img.copy()

    id = 0
    tiles = {}

    for y in range(0, new_height, tile_h):
        for x in range(0, new_width, tile_w):
            cv2.rectangle(mask, pt1=(x,y), pt2=(x+tile_w-1,y+tile_h-1), color=COLOR, thickness=1)
            cv2.putText(mask, str(id), (x,y+tile_h), FONT, 1, COLOR, 3, cv2.LINE_AA)
            tile = img[y:y+tile_h, x:x+tile_w]
            tiles[id] = tile
            id = id + 1

    return img, tiles, mask

def glcmXbyXWinScan(sarraster, windowSize):
    #Create rasters to receive texture and define filenames
    contrastraster = np.zeros((sarraster.shape[0], sarraster.shape[1]), dtype = float)
    contrastraster[:] = 0.0
    
    dissimilarityraster = np.zeros((sarraster.shape[0], sarraster.shape[1]), dtype = float)
    dissimilarityraster[:] = 0.0
    
    #homogeneityraster = np.copy(sarraster)
    homogeneityraster = np.zeros((sarraster.shape[0], sarraster.shape[1]), dtype = float)
    homogeneityraster[:] = 0.0
    
    #energyraster = np.copy(sarraster)
    energyraster = np.zeros((sarraster.shape[0], sarraster.shape[1]), dtype = float)
    energyraster[:] = 0.0
    
    #correlationraster = np.copy(sarraster)
    correlationraster = np.zeros((sarraster.shape[0], sarraster.shape[1]), dtype = float)
    correlationraster[:] = 0.0
    
    #ASMraster = np.copy(sarraster)
    ASMraster = np.zeros((sarraster.shape[0], sarraster.shape[1]), dtype = float)
    ASMraster[:] = 0.0
    
    
    for i in range(sarraster.shape[0]):
        for j in range(sarraster.shape[1]):
            if i < windowSize or j < windowSize:
                continue
            if i > (contrastraster.shape[0] - windowSize) or j > (contrastraster.shape[1] - windowSize):
                continue
            
            # Define size of moving window
            glcm_window = sarraster[i-windowSize: i+windowSize, j-windowSize : j+windowSize]
            
            # Calculate GLCM and textures
            glcm = greycomatrix(glcm_window, [1], [0],  symmetric = True, normed = True )
    
            # Calculate texture and write into raster where moving window is centered
            contrastraster[i,j]      = greycoprops(glcm, 'contrast')
            dissimilarityraster[i,j] = greycoprops(glcm, 'dissimilarity')
            homogeneityraster[i,j]   = greycoprops(glcm, 'homogeneity')
            energyraster[i,j]        = greycoprops(glcm, 'energy')
            correlationraster[i,j]   = greycoprops(glcm, 'correlation')
            ASMraster[i,j]           = greycoprops(glcm, 'ASM')
            glcm = None
            glcm_window = None
            
    #Normalization use when only needed   
    contrastraster = 255.0 * normalize(contrastraster)
    contrastraster = contrastraster.astype(int)
    
    dissimilarityraster = 255.0 * normalize(dissimilarityraster)
    dissimilarityraster = dissimilarityraster.astype(int)
    
    homogeneityraster = 255.0 * normalize(homogeneityraster)
    homogeneityraster = homogeneityraster.astype(int)
    
    energyraster = 255.0 * normalize(energyraster)
    energyraster = energyraster.astype(int)
    
    correlationraster = 255.0 * normalize(correlationraster)
    correlationraster = correlationraster.astype(int)
    
    ASMraster = 255.0 * normalize(ASMraster)
    ASMraster = ASMraster.astype(int)

    return contrastraster, dissimilarityraster, homogeneityraster, energyraster, correlationraster, ASMraster


def normalize(arrayX):
    for i in range(arrayX.shape[0]):
        for j in range(arrayX.shape[1]):
            arrayX[i,j] = ((arrayX[i,j] - arrayX.min()) / (arrayX.max() - arrayX.min()))
            
    return arrayX


def display_window_glcm(img):
    img = img[:,:,0]

    windowSz = 15       # change the window size based on need
    contrastScan, dissimilarityScan, homogeneityScan, energyScan, correlationScan, ASMScan = glcmXbyXWinScan(img, windowSz)

    fig = plt.figure(figsize=(22, 18))
    texturelist = {1: 'contrast', 2: 'dissimilarity', 3: ' homogeneity', 4: 'energy', 5: 'correlation', 6: 'ASM'}
    for key in texturelist:
        ax = plt.subplot(3,2,key)
        plt.axis('off')
        ax.set_title(texturelist[key])
        plt.imshow(eval(texturelist[key] + "Scan"), cmap = 'gray')

    plt.show()


def display_glcm(img):
    img = img[:,:,0]

    glcm = greycomatrix(img, [1], [0],  symmetric = True, normed = True )
    contrastraster = greycoprops(glcm, 'contrast')
    dissimilarityraster = greycoprops(glcm, 'dissimilarity')
    homogeneityraster = greycoprops(glcm, 'homogeneity')
    energyraster = greycoprops(glcm, 'energy')
    correlationraster = greycoprops(glcm, 'correlation')
    ASMraster = greycoprops(glcm, 'ASM')

    fig = plt.figure(figsize=(22, 18))
    texturelist = {1: 'contrast', 2: 'dissimilarity', 3: ' homogeneity', 4: 'energy', 5: 'correlation', 6: 'ASM'}
    for key in texturelist:
        ax = plt.subplot(3,2,key)
        plt.axis('off')
        ax.set_title(texturelist[key])
        plt.imshow(eval(texturelist[key] + "raster"), cmap = 'gray')

    plt.show()