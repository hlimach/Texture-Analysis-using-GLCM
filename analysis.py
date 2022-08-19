import cv2
import bisect
from matplotlib import colors
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from glcm_helpers import FEATURES

FONT = cv2.FONT_HERSHEY_DUPLEX

def scatterplot_labelled_data(avg_tile_features, tile_labels):
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

        n, bins, patches = plt.hist([y0,y1], bin_count, color=colors, label=labels)
        plt.xticks(bins)
        plt.legend()
    plt.show()

def analyze_histogram(img, tiles, tile_labels, avg_tile_features, bin_count=10):
    colors = ('black', 'red')
    labels = ('Pavement', 'Pothole')

    for key in FEATURES:
        # add histogram with titled bins
        fig = plt.figure(figsize=(22, 18))
        a0, a1 = fig.subplots(2, 1, gridspec_kw={'height_ratios': [1, 4]})

        y0 = [avg_tile_features[id][FEATURES[key]] for id in tile_labels if tile_labels[id]==labels[0]]
        y1 = [avg_tile_features[id][FEATURES[key]] for id in tile_labels if tile_labels[id]==labels[1]]

        n, bins, patches = a0.hist([y0,y1], bin_count, color=colors, label=labels)
        additive = (bins[2] - bins[1]) / 2
        ticks = [bins[i]+additive for i in range(bin_count)]
        ticklabels = ['bin '+str(i) for i in range(bin_count)]

        a0.set_title(FEATURES[key])
        a0.set_xticks(ticks)
        a0.set_xticklabels(ticklabels, rotation=90)
        a0.legend()

        # add image with histogram labelled tiles
        grid = img.copy()
        height, width = grid.shape[:2]
        h, w = tiles[1].shape[:2]
        id = 0

        for y in range(0, height, h):
            for x in range(0, width, w):
                # find out which bin the tile belongs to
                bin_num = bisect.bisect(bins, avg_tile_features[id][FEATURES[key]]) - 1

                # if it is pavement, draw black rect and add black text, otherwise red
                if tile_labels[id] == labels[0]:
                    cv2.rectangle(grid, pt1=(x,y), pt2=(x+w-1,y+h-1), color=(0,0,0), thickness=1)
                    cv2.putText(grid, str(bin_num), (x,y+h), FONT, 1, (0,0,0), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(grid, pt1=(x,y), pt2=(x+w-1,y+h-1), color=(0,0,255), thickness=1)
                    cv2.putText(grid, str(bin_num), (x,y+h), FONT, 1, (0,0,255), 1, cv2.LINE_AA)
                id = id + 1

        a1.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))

    plt.show()

# takes a feature value and the bins to be shown
# bins in different colors, but potholes remain red
# outputs histogram and image like above
# only tiled parts are specified bins

def analyze_feature_by_bin(feature, target_colored_bins, img, tiles, avg_tile_features, bin_count=10):
    # add histogram with titled bins
    fig = plt.figure(figsize=(22, 18))
    a0, a1 = fig.subplots(2, 1, gridspec_kw={'height_ratios': [1, 4]})

    y = [avg_tile_features[id][feature] for id in avg_tile_features]
    n, bins, patches = a0.hist(y, bin_count)

    for i in range(len(patches)):
        if i in target_colored_bins.keys():
            patches[i].set_facecolor(target_colored_bins[i])
        else:
            patches[i].set_facecolor('black')

    additive = (bins[2] - bins[1]) / 2
    ticks = [bins[i]+additive for i in range(bin_count)]
    ticklabels = ['bin '+str(i) for i in range(bin_count)]

    a0.set_title(feature)
    a0.set_xticks(ticks)
    a0.set_xticklabels(ticklabels, rotation=90)

    # add image with target tiles
    grid = img.copy()
    height, width = grid.shape[:2]
    h, w = tiles[1].shape[:2]
    id = 0

    for y in range(0, height, h):
        for x in range(0, width, w):
            # find out which bin the tile belongs to
            bin_num = bisect.bisect(bins, avg_tile_features[id][feature]) - 1

            if bin_num in target_colored_bins.keys():
                rgb = colors.to_rgb(target_colored_bins[bin_num])
                bgr = (rgb[2]*255, rgb[1]*255, rgb[0]*255)
                cv2.rectangle(grid, pt1=(x,y), pt2=(x+w-1,y+h-1), color=bgr, thickness=2)
                cv2.putText(grid, str(bin_num), (x,y+h), FONT, 1, bgr, 2, cv2.LINE_AA)
            id = id + 1
    a1.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
    plt.show()
