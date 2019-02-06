

def otsuThreshold(img):
    img = img.astype(np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    global_thresh = threshold_otsu(img_gray)
    binary_global = img_gray < global_thresh

    temp = morphology.remove_small_objects(
        binary_global, min_size=500, connectivity=1)
    mask = morphology.remove_small_holes(temp, 500, connectivity=2)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for i in range(len(img_bgr)):
        for j in range(len(img_bgr[0])):
            if mask[i][j] == False:
                img_bgr[i][j][0] = 0
                img_bgr[i][j][1] = 0
                img_bgr[i][j][2] = 0

    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    return mask, img_bgr.astype(np.uint64)
