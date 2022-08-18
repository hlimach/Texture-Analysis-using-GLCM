import os
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (255,0,0)

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
            
            cv2.rectangle(grid, pt1=(x,y), pt2=(x+w-1,y+h-1), color=COLOR, thickness=1)
            cv2.putText(grid, str(id), (x,y+h), FONT, 1, COLOR, 1, cv2.LINE_AA)
            id = id + 1

    return img, tiles, labels, grid

