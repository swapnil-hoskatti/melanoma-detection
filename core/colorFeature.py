#####################################
#   Colour Features Extraction      #
#       colour covariance           #
#####################################

# imports - final imports
from . import np


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
    return bg[0][0], bg[0][1], br[0][1], bg[1][1], br[0][1], gr[0][1], gr[1][1]


def adhoc(m, image, avgradius, ratio, c):
    central = [0, 0, 0]
    ac = 0
    ab = 0
    border = [0, 0, 0]
    cx = c[0]
    cy = c[1]
    for i in range(0, 450):
        for j in range(0, 600):
            a = avgradius*ratio
            if(m[i][j] == True):
                if(np.linalg.norm(np.array([cx, cy])-np.array([j, i])) <= a):
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
    count_pixels = ac+ab

    try:
        central = [i/ac for i in central]
        border = [i/ab for i in border]

        adhoc = [0, 0, 0]
        adhoc[0] = central[0]/border[0]
        adhoc[1] = central[1]/border[1]
        adhoc[2] = central[2]/border[2]

        return adhoc[0], adhoc[1], adhoc[2]
    except:
        return np.nan, np.nan, np.nan


def ColorFeatures(mask, image, avgRadius, c):
    c_bb, c_bg, c_br, c_gg, c_br, c_gr, c_rr = covariance(mask, image)
    adhocb1, adhocg1, adhocr1 = adhoc(mask, image, avgRadius, 1/3, c)
    adhocb2, adhocg2, adhocr2 = adhoc(mask, image, avgRadius, 9/10, c)

    return c_bb, c_bg, c_br, c_gg, c_br, c_gr, c_rr, adhocb1, adhocg1, adhocr1, adhocb2, adhocg2, adhocr2
