#####################################
#   Texture Features Extraction     #
# irregularity, circularity, radius #
#####################################

# imports - final imports
from . import feature, np


def GeometricFeatures(mask):
    try:
        edges1 = feature.canny(mask, sigma=0)
    except:
        return (np.nan,) * 7

    perimeterpixels = 0
    areapixels = 0

    for l in mask:
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
            if mask[i][j] == True:

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
