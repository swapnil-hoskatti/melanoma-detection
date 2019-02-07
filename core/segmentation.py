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
