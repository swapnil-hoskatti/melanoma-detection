#####################################
#           Segmentation            #
#         generation of ROI         #
#####################################

# imports - final imports
from . import cv2, morphology, np, threshold_otsu, keras, sqrt


def unetSegment(img):
    """
    description: unet segmentation
    params: np.ndarray -> uint8 :: values = (0, 255)
    returns: np.ndarray -> float32 :: values = (0, 1)
    """
    img = np.expand_dims(img, axis=0)
    CLASSIFICATION_MODEL_ARCH_PATH = "../core/models/unet-segment.json"
    CLASSIFICATION_MODEL_WEIGHTS_PATH = "../core/models/unet-segment.h5"

    json_file = open(CLASSIFICATION_MODEL_ARCH_PATH, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)

    model.load_weights(CLASSIFICATION_MODEL_WEIGHTS_PATH)
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    unet_mask = model.predict([img])[0]

    return unet_mask


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
            try:
                if mask[i][j] == 0:
                    img[i][j] = (0, 0, 0)
            except ValueError:
                if any(mask[i][j]) == 0:
                    img[i][j] = (0, 0, 0)
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def mainBlob(mask):
    """
    description: finds blob closest to the center of image
    params:
    returns:
    """

    temp, c = [[[0, 0, 0] for x in range(0, 600)] for y in range(0, 450)], 0
    
    # Extraction of lighter lesions
    for i, n in enumerate(mask):
        for j, m in enumerate(n):
            if any(m):
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
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    mask = np.zeros(mask.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, saved_contour, (255, 255, 255), -1)
    h, w = mask.shape[:2]
    if (mask[c[1]][c[0]] == 255):
        res = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(mask, res, c, 255)
    else :
        res = np.ones((h + 2, w + 2), np.uint8)
        cv2.floodFill(mask, res, c, 0)
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((7, 7), np.uint8)
    dilated_img = cv2.dilate(mask, kernel, iterations=2)
    result = cv2.morphologyEx(dilated_img, cv2.MORPH_CLOSE, kernel).astype(np.bool_)

    return result
