#####################################
#           Segmentation            #
#         generation of ROI         #
#####################################

# imports - final imports
from . import cv2, morphology, np, threshold_otsu, keras


def unetSegment(img):
    img = np.expand_dims(img, axis=0)
    CLASSIFICATION_MODEL_ARCH_PATH = "../core/models/unet-segment.json"
    CLASSIFICATION_MODEL_WEIGHTS_PATH = "../core/models/unet-segment.h5"

    json_file = open(CLASSIFICATION_MODEL_ARCH_PATH, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)

    model.load_weights(CLASSIFICATION_MODEL_WEIGHTS_PATH)
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    return model.predict([img])[0]


def otsuThreshold(img):
    img = img.astype(np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    global_thresh = threshold_otsu(img_gray)
    binary_global = img_gray < global_thresh

    temp = morphology.remove_small_objects(binary_global, min_size=500, connectivity=1)
    mask = morphology.remove_small_holes(temp, 500, connectivity=2)

    return mask


def getROI(img, mask):
    # maps mask with img to generate ROI
    for i in range(len(img)):
        for j in range(len(img[0])):
            if any(mask[i][j]) == 0:
                img[i][j] = (0, 0, 0)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def mainBlob(image):
    ### takes input as "combinedSegmented" image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mindist = 999.9
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"]!=0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            dist = math.sqrt((cX-299)**2+(cY-224)**2)
            if mindist > dist:
                saved_contour = i
                mindist = dist
                c = (cX,cY)
    mask = np.zeros(image.shape, np.uint8)
    result = cv2.drawContours(mask, contours, saved_contour, (255, 255, 255), -1)

    h, w = result.shape[:2]
    res = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(result, res, c, 255)

    kernel = np.ones((7,7), np.uint8)
    result = cv2.dilate(result, kernel, iterations=2)

    return result