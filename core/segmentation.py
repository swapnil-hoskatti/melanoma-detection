#####################################
#           Segmentation            #
#         generation of ROI         #
#####################################

# imports - final imports
from . import cv2, morphology, np, threshold_otsu, keras

FIRST_LOAD = 0

def unetSegment(img):
    img = np.expand_dims(img, axis=0)
    CLASSIFICATION_MODEL_ARCH_PATH = '../core/models/unet-segment.json'
    CLASSIFICATION_MODEL_WEIGHTS_PATH = '../core/models/unet-segment.h5'

    json_file = open(CLASSIFICATION_MODEL_ARCH_PATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)

    model.load_weights(CLASSIFICATION_MODEL_WEIGHTS_PATH)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    
    return model.predict([img])[0]

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

def largestBlobFinder(image):
    #takes input as "combinedSegmented" image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_Area = 0
    for i,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_Area:
            max_Area = area
            saved_contour = i

    mask = np.zeros(image.shape, np.uint8)
    result = cv2.drawContours(mask,contours,saved_contour,(255,255,255),-1)
    result_copy = result.copy()
    x,y = [],[]
    for p,k in enumerate(result):
        for q,j in enumerate(k):
            if j == 255:
                x.append(p)
                y.append(q)
    u,d,l,r=min(x),max(x),min(y),max(y)
    c = (int((u+d)/2),int((l+r)/2))

    h,w = result.shape[:2]
    res = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(result, res, c, 255)
    result = result | result_copy
    
    #if the largest blob found is exactly opposite of the ROI
    if int(result[225][300]) != 255:
        result = cv2.bitwise_not(result)

    kernel = np.ones((7,7),np.uint8)
    result = cv2.dilate(result,kernel,iterations = 2)

    return result