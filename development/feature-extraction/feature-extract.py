# imports - standard imports
import multiprocessing
import os
import random
from collections import Counter
from copy import copy, deepcopy
from math import sqrt
from statistics import mean as st_mean
from statistics import median
from statistics import mode as st_mode
from statistics import stdev as st_stdev

# imports - third party imports
import cv2
import numpy as np
from scipy import ndimage, stats
from skimage import data, feature, filters, io, morphology
from skimage.filters import threshold_adaptive, threshold_otsu
from sklearn.cluster import KMeans


def otsuThreshold(img):
    """
    description: otsu's + modifications to remove lighter skin lesions
    params: np.ndarray -> uint8 :: values = (0, 255)
    returns:
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    global_thresh = threshold_otsu(img_gray)
    binary_global = img_gray < global_thresh

    temp = morphology.remove_small_objects(binary_global, min_size=500, connectivity=1)
    mask = morphology.remove_small_holes(temp, 500, connectivity=2)

    temp, c = [[[0, 0, 0] for x in range(0, 600)] for y in range(0, 450)], 0

    # Extraction of lighter lesions
    for i, n in enumerate(mask):
        for j, m in enumerate(n):
            if m:
                temp[i][j] = [255, 255, 255]
                c = c - 1
            else:
                c = c + 1
    if c < 0:
        for p, i in enumerate(temp):
            for q, j in enumerate(i):
                if j == [255, 255, 255]:
                    temp[p][q] = [0, 0, 0]
                else:
                    temp[p][q] = [255, 255, 255]
    mask = np.array(temp, dtype=np.uint8)

    return mask


def getROI(img, mask):
    """
    description: maps mask with img to generate ROI
    params:
        img:
        mask:
    returns:
    """

    for i in range(len(img)):
        for j in range(len(img[0])):
            if mask[i][j] == 0:
                img[i][j] = (0, 0, 0)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def mainBlob(image):
    """
    description: finds blob closest to the center of image
    params:
    returns:
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mindist = 999.9

    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            dist = sqrt((cX - 299) ** 2 + (cY - 224) ** 2)
            if mindist > dist:
                saved_contour = i
                mindist = dist
                c = (cX, cY)
    mask = np.zeros(image.shape, np.uint8)
    result = cv2.drawContours(mask, contours, saved_contour, (255, 255, 255), -1)

    h, w = result.shape[:2]
    res = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(result, res, c, 255)

    kernel = np.ones((7, 7), np.uint8)
    dilated_img = cv2.dilate(result, kernel, iterations=2)
    result = cv2.morphologyEx(dilated_img, cv2.MORPH_CLOSE, kernel).astype(np.bool_)

    return result


def covariance(mask, img):
    b, g, r, m, count, icount = [], [], [], [], 0, 0
    for i, t in enumerate(mask):
        for j, s in enumerate(t):
            if s == True:
                m.append(img[i][j])
                b.append(img[i][j][0])
                g.append(img[i][j][1])
                r.append(img[i][j][2])
    bg, br, gr = np.cov(b, g), np.cov(b, r), np.cov(g, r)

    #   B-B, B-G, B-R, G-G, G-R, R-R
    return bg[0][0], bg[0][1], br[0][1], bg[1][1], gr[0][1], gr[1][1]


def adhoc(m, image, avgradius, ratio, c):
    central = [0, 0, 0]
    ac = 0
    ab = 0
    border = [0, 0, 0]
    cx = c[0]
    cy = c[1]
    for i in range(0, 450):
        for j in range(0, 600):
            a = avgradius * ratio
            if m[i][j] == True:
                if np.linalg.norm(np.array([cx, cy]) - np.array([j, i])) <= a:
                    central[0] += image[i][j][0]
                    central[1] += image[i][j][1]
                    central[2] += image[i][j][2]
                    ac += 1
                else:
                    border[0] += image[i][j][0]
                    border[1] += image[i][j][1]
                    border[2] += image[i][j][2]
                    ab += 1

    total = [sum(x) for x in zip(central, border)]
    count_pixels = ac + ab

    try:
        central = [i / ac for i in central]
        border = [i / ab for i in border]

        adhoc = [0, 0, 0]
        adhoc[0] = central[0] / border[0]
        adhoc[1] = central[1] / border[1]
        adhoc[2] = central[2] / border[2]

        return adhoc[0], adhoc[1], adhoc[2]
    except:
        return np.nan, np.nan, np.nan


def ColorFeatures(mask, image, avgRadius, c):
    c_bb, c_bg, c_br, c_gg, c_gr, c_rr = covariance(mask, image)
    adhocb1, adhocg1, adhocr1 = adhoc(mask, image, avgRadius, 1 / 3, c)
    adhocb2, adhocg2, adhocr2 = adhoc(mask, image, avgRadius, 9 / 10, c)

    return (
        c_bb,
        c_bg,
        c_br,
        c_gg,
        c_gr,
        c_rr,
        adhocb1,
        adhocg1,
        adhocr1,
        adhocb2,
        adhocg2,
        adhocr2,
    )


def GeometricFeatures(m):
    edges1 = feature.canny(m, sigma=0)

    perimeterpixels = 0
    areapixels = 0

    for l in m:
        for p in l:
            if p == True:
                areapixels += 1

    # areapixels/=10
    a = areapixels
    for l in edges1:
        for p in l:
            if p == True:
                perimeterpixels += 1

    # perimeterpixels/=10
    p = perimeterpixels
    # CIRCULARITY INDEX
    crc = (4 * areapixels * 3.14) / (perimeterpixels * perimeterpixels)

    # IRREGULARITY INDEX A
    ira = p / a

    sx = 0
    sy = 0
    count = 0
    for i in range(0, 450):
        for j in range(0, 600):
            if m[i][j] == True:

                sy += i
                sx += j
                count += 1
    cx = int(sx / count)
    cy = int(sy / count)

    c = [cx, cy]

    x = []
    y = []
    for i in range(0, 450):
        for j in range(0, 600):
            if edges1[i][j] == True:
                y.append(i)
                x.append(j)
    maxl = 9999
    minl = 0

    x = np.array(x) - cx
    y = np.array(y) - cy
    count = 0
    radii = 0
    min_r = 99999
    max_r = 0
    for i in range(0, len(x)):
        dist = np.linalg.norm(np.array([0, 0]) - np.array([x[i], y[i]]))
        radii += dist
        count += 1

        if dist < min_r and dist != 0:
            min_r = dist
        if dist > max_r:
            max_r = dist

    avgradius = radii / count

    gd = 2 * max_r
    sd = 2 * min_r

    irb = p / gd

    irc = p * (1 / sd - 1 / gd)

    ird = gd - sd

    return crc, ira, irb, irc, ird, avgradius, c


ROUND_FACTOR = 4

def __histogram(img):
    b_, g_, r_ = cv2.split(img)

    H_b, H_g, H_r = {}, {}, {}

    for pixel in b_.flatten():
        if pixel not in H_b.keys():
            H_b[pixel] = 1
        else:
            H_b[pixel] += 1

    for pixel in g_.flatten():
        if pixel not in H_g.keys():
            H_g[pixel] = 1
        else:
            H_g[pixel] += 1

    for pixel in r_.flatten():
        if pixel not in H_r.keys():
            H_r[pixel] = 1
        else:
            H_r[pixel] += 1

    return H_b, H_g, H_r


def __centerGravity(H_b, H_g, H_r):
    return median(H_b), median(H_g), median(H_r)


def texture(img, mask):
    bins = {
        "0 0 0": None,
        "0 0 1": None,
        "0 1 0": None,
        "1 0 0": None,
        "1 1 0": None,
        "0 1 1": None,
        "1 0 1": None,
        "1 1 1": None,
    }

    H_b, H_g, H_r = __histogram(img)
    b_median, g_median, r_median = __centerGravity(H_b, H_g, H_r)

    for y, rows in enumerate(img):
        for x, pixel in enumerate(rows):
            if any(mask[y][x] != [0, 0, 0]):
                for index, value in enumerate(pixel):
                    if value is not None:
                        if index == 0:
                            # 0 corresponds to B plane in openCV split
                            if value > b_median:
                                t_b = 1
                            else:
                                t_b = 0

                        if index == 1:
                            # 1 corresponds to G plane in openCV split
                            if value > g_median:
                                t_g = 1
                            else:
                                t_g = 0

                        if index == 2:
                            # 2 corresponds to R plane in openCV split
                            if value > r_median:
                                t_r = 1
                            else:
                                t_r = 0

                bin_val = f"{t_b} {t_g} {t_r}"

                if bins[bin_val] == None:
                    bins[bin_val] = [img[y][x]]
                else:
                    bins[bin_val] += [img[y][x]]

    return bins


def mean(*args):
    return tuple(np.mean(x) for x in args)


def mode(*args):
    return tuple(stats.mode(x) for x in args)


def std_dev(*args):
    return tuple(np.std(x) for x in args)


def skewness(mean, mode, std_dev):
    return (mean - mode) / std_dev


def kurlosis():
    return


def features(bins):
    """
    input bins dict which contains name of bin and posistion of pixel : (x,y)
    """
    all_features = tuple()

    for bin_name, pixels in bins.items():
        if pixels:
            Bmean = mean([x[0] for x in pixels])[0]
            Gmean = mean([x[1] for x in pixels])[0]
            Rmean = mean([x[2] for x in pixels])[0]

            # dont round it as its not directly used
            Bmode = (mode([x[0] for x in pixels])[0])[0]
            Gmode = (mode([x[1] for x in pixels])[0])[0]
            Rmode = (mode([x[2] for x in pixels])[0])[0]

            Bstd = std_dev([x[0] for x in pixels])[0]
            Gstd = std_dev([x[1] for x in pixels])[0]
            Rstd = std_dev([x[2] for x in pixels])[0]

            Bsk = skewness(Bmean, Bmode, Bstd)[0]
            Gsk = skewness(Gmean, Gmode, Gstd)[0]
            Rsk = skewness(Rmean, Rmode, Rstd)[0]
        
        else:
            Bmean = np.nan
            Gmean = np.nan
            Rmean = np.nan

            Bstd = np.nan
            Gstd = np.nan
            Rstd = np.nan

            Bsk = np.nan
            Gsk = np.nan
            Rsk = np.nan

        all_features += Bmean, Gmean, Rmean, Bstd, Gstd, Rstd, Bsk, Gsk, Rsk

    return all_features


def TextureFeatures(mask, img):
    bins = texture(img, mask)
    return features(bins)


# image acquisition
def rsize(img):
    """
    description: resizing to fixed dimens
    params: np.ndarray -> uint8 :: values = (0, 255)
    returns: np.ndarray -> uint8 :: values = (0, 255)
    """
    if img.shape != (450, 600, 3):
        # REF: https://stackoverflow.com/a/48121983/10309266
        return cv2.resize(img, dsize=(600, 450), interpolation=cv2.INTER_CUBIC)

    return img


def read(path):
    """
    description: image acquisition
    params: np.ndarray -> uint8 :: values = (0, 255)
    returns: np.ndarray -> uint8 :: values = (0, 255)
    """
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img.any():
            return rsize(img)
        else:
            return "Oops!"
    else:
        return FileNotFoundError


def extract(img_name):
    img_ = img_name
    img_name = os.path.join(DIR_NAME, img_name)
    img = read(img_name)
    otsu_mask = otsuThreshold(img)

    mask = mainBlob(np.array(otsu_mask, dtype=np.uint8))
    roi = getROI(img, np.array(mask))
    io.imsave(f"seg/{img_}", roi)

    crc, ira, irb, irc, ird, avgRadius, c = GeometricFeatures(mask)
    c_bb, c_bg, c_br, c_gg, c_gr, c_rr, adhocb1, adhocg1, adhocr1, adhocb2, adhocg2, adhocr2 = ColorFeatures(
        mask, img, avgRadius, c
    )
    # 8 * 9 => 8 * (Bmean, Gmean, Rmean, Bstd, Gstd, Rstd, Bsk, Gsk, Rsk) 
    texture_features = TextureFeatures(mask, img)
    
    with open("features-90.csv", "a") as f:
        f.write(
            f"{img_}"
            + "".join([f"{x}," for x in texture_features])
            + f"{crc},{ira},{irb},{irc},{ird},{c_bb},{c_bg},{c_br},{c_gg},{c_gr},{c_rr},{adhocb1},{adhocg1},{adhocr1},{adhocb2},{adhocg2},{adhocr2}\n"
        )


if __name__ == "__main__":
    DIR_NAME = "img"
    num_processors = multiprocessing.cpu_count()
    imgs = os.listdir(DIR_NAME)

    with open("features-90.csv", "w") as f:
        f.write(
            "img_id"
            + "".join([f"{x}_{i}," for i in range(1,9) for x in ["Bmean", "Gmean", "Rmean", "Bstd", "Gstd", "Rstd", "Bsk", "Gsk", "Rsk"]])
            + f"{crc},{ira},{irb},{irc},{ird},{c_bb},{c_bg},{c_br},{c_gg},{c_gr},{c_rr},{adhocb1},{adhocg1},{adhocr1},{adhocb2},{adhocg2},{adhocr2}\n"
        )

    with multiprocessing.Pool(processes=num_processors) as pool:
        pool.map(extract, imgs)
