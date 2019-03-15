#################################################################################
#                       Main Application - Base System                          #
#   image-acquisition -> segmentation -> Feature Extraction -> Classification   #
#################################################################################

# imports - imports for segmentation
from core.segmentation import otsuThreshold, unetSegment, mainBlob, getROI

# imports - imports for feature extraction
from core.colorFeature import ColorFeatures
from core.textureFeature import TextureFeatures
from core.geometricFeature import GeometricFeatures
from core.clean import remove_hair

# imports - final imports
from . import os, cv2, io, keras, np, morphology

# constants - global constants for classification
DNN_CLASSIFICATION_MODEL_ARCH_PATH = "../core/models/deep-classify.json"
DNN_CLASSIFICATION_MODEL_WEIGHTS_PATH = "../core/models/deep-classify.h5"
CNN_CLASSIFICATION_MODEL_ARCH_PATH = "../core/models/vgg_16.json"
CNN_CLASSIFICATION_MODEL_WEIGHTS_PATH = "../core/models/vgg_16-model-val_acc-76-acc-80.hdf5"

# global constants
ROUND_FACTOR = 4

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
            print("Stage 0: Acquisition Done")
            return rsize(img)
        else:
            return "Oops!"
    else:
        return FileNotFoundError


def procedure(img):
    """
    description: main algorithm
    params: np.ndarray -> uint8 :: values = (0, 255)
    returns: int (label data) :: values = (0, 6)
    """
    # pre processing
    hair_rem = remove_hair(img)
    io.imsave("../temp_files/hairOG.jpg", hair_rem)

    # segmentation
    unet_mask = cv2.cvtColor(unetSegment(hair_rem), cv2.COLOR_GRAY2BGR)
    io.imsave("../temp_files/unetSegmentOG.jpg", unet_mask)
    unet_mask = unet_mask.astype(np.uint8)

    otsu_mask = otsuThreshold(hair_rem) * 255
    io.imsave("../temp_files/otsuSegmentOG.jpg", otsu_mask)

    temp = [[[0, 0, 0] for x in range(0, 600)] for y in range(0, 450)]

    # combine unet and otsu's mask
    for i in range(450):
        for j in range(600):
            otsu = otsu_mask[i][j]
            unet = unet_mask[i][j]
            if any(unet == [1, 1, 1]) or any(otsu == [255, 255, 255]):
                temp[i][j] = [255] * 3
            else:
                temp[i][j] = [0] * 3
    io.imsave("../temp_files/combinedSegmentOG.jpg", np.array(temp))

    # mask.dtype -> bool
    mask = mainBlob(np.array(temp, dtype=np.uint8))
    io.imsave("../temp_files/mainBlobOG.jpg", np.array(mask, dtype=np.uint8) * 255)

    roi = getROI(img, np.array(mask))
    io.imsave("../temp_files/ROI.jpg", roi)

    print("Stage 1: Segmentation Done")

    # feature extraction
    crc, ira, irb, irc, ird, avgRadius, c = GeometricFeatures(mask)
    c_bb, c_bg, c_br, c_gg, c_gr, c_rr, adhocb1, adhocg1, adhocr1, adhocb2, adhocg2, adhocr2 = ColorFeatures(
        mask, img, avgRadius, c
    )
    # textureFeatures return a tuple of 8 * 9
    texture_features = list(TextureFeatures(mask, img))
    print("Stage 2: Feature Extraction Done")

    features = [
        texture_features
            +
        [
            crc,
            ira,
            irb,
            irc,
            ird,
            c_bb,
            c_bg,
            c_br,
            c_gg,
            c_br,
            c_gr,
            c_rr,
            adhocb1,
            adhocg1,
            adhocr1,
            adhocb2,
            adhocg2,
            adhocr2,
        ]
    ]

    # Only for testing:
    print([round(x, ROUND_FACTOR) for x in features[0]])
    return
    
    # DNN here
    loaded_model_json = open(DNN_CLASSIFICATION_MODEL_ARCH_PATH).read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(DNN_CLASSIFICATION_MODEL_WEIGHTS_PATH)
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    score = int(model.predict([features])[0][0])
    print(f"Stage 3: Prediction Done ~ {score}")

    # CNN here
    json_file = open(CNN_CLASSIFICATION_MODEL_ARCH_PATH).read()
    model_cnn = keras.models.model_from_json(json_file)
    model_cnn.load_weights(CNN_CLASSIFICATION_MODEL_WEIGHTS_PATH)
    model_cnn.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    score = int(model_cnn.predict_classes([
        cv2.resize(
                roi,
                dsize=(256, 256),
                interpolation=cv2.INTER_CUBIC
            ).reshape((1, 256, 256, 3))
        ]))
    print(f"Stage 3: CNN Prediction Done ~ {score}")
    # 5 - VASC ??? everything gets predicted as 5. needs more testing
    print("---"*10)
    
    return score


def main_app(path):
    """
    description: importable to FLASK app
    params: str (path)
    returns: int (label) :: values = (0, 6)
    """
    img = read(path)
    return procedure(img)
