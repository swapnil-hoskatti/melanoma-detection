#####################################
#   Texture Features Extraction     #
#       Bins: algorithm             #
#####################################


def __histogram(img):
    """
    generating hist for R, G, B planes
    returns: (b,g,r) hist dict tuple
            ({0:12,...,255:100},{0:12,...,255:100},{0:12,1:33,...,255:100})
    """

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


def texture(img):
    """
    Using bins algorithm,
    texture features are obtained and moments are generated

    returns: 8 bins dict
            {000:something, 001:something, 100:something, 010:something,...}
    """
    bins = {}

    H_b, H_g, H_r = __histogram(img)
    b_median, g_median, r_median = __centerGravity(H_b, H_g, H_r)

    for y, rows in enumerate(img):
        for x, pixel in enumerate(rows):
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

# Multiple inputs, can take multiple arrays


def mean(*args):
    return tuple(np.mean(x) for x in args)


def mode(*args):
    return tuple(stats.mode(x) for x in args)


def std_dev(*args):
    #  σ
    return tuple(np.std(x) for x in args)


# single arrays' data only

def skewness(mean, mode, std_dev):
    return (mean - mode) / std_dev


def kurtosis():
    return


def TextureFeatures(bins):
    """
    input bins dict which contains name of bin and posistion of pixel : (x,y)
    """

    for bin_name, pixels in bins.items():
        # print(bin_name, "\t", len(pixels))

        Bmean = round(mean([x[0] for x in pixels])[0], 2)
        Gmean = round(mean([x[1] for x in pixels])[0], 2)
        Rmean = round(mean([x[2] for x in pixels])[0], 2)

        Bmode = (mode([x[0] for x in pixels])[0], 2)
        Gmode = (mode([x[1] for x in pixels])[0], 2)
        Rmode = (mode([x[2] for x in pixels])[0], 2)

        Bstd = round(std_dev([x[0] for x in pixels])[0], 2)
        Gstd = round(std_dev([x[1] for x in pixels])[0], 2)
        Rstd = round(std_dev([x[2] for x in pixels])[0], 2)

        return Bmean, Gmean, Rmean, Bstd, Gstd, Rstd
