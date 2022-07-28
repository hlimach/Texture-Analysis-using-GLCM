import os
import cv2

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
            cv2.putText(mask, str(id), (x,y+25), FONT, 1, COLOR, 3, cv2.LINE_AA)
            tile = img[y:y+tile_h, x:x+tile_w]
            tiles[id] = tile
            id = id + 1

    return img, tiles, mask

