#####################################
#   Texture Features Extraction     #
#       Bins: algorithm             #
#####################################

# imports - final imports
from . import median, np, stats, cv2


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
    bins = {}

    H_b, H_g, H_r = __histogram(img)
    b_median, g_median, r_median = __centerGravity(H_b, H_g, H_r)

    for y, rows in enumerate(img):
        for x, pixel in enumerate(rows):
            if any(mask[y][x] != [0, 0, 0]):
                for index, value in enumerate(pixel):
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

                if bin_val not in bins.keys():
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

        all_features += Bmean, Gmean, Rmean, Bstd, Gstd, Rstd, Bsk, Gsk, Rsk
    
    return all_features


def TextureFeatures(mask, img):
    bins = texture(img, mask)
    return features(bins)
