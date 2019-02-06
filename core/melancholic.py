#################################################################################
#                       Main Application - Base System                          #
#   image-acquisition -> segmentation -> Feature Extraction -> Classification   #
#################################################################################

# imports - imports for segmentation
from segmentation import otsuThreshold

# imports - imports for feature extraction
from colorFeature import ColorFeatures
from textureFeature import TextureFeatures
from geometricFeature import GeometricFeatures

# imports - imports for classification
import keras


# global constants
CLASSIFICATION_MODEL_ARCH_PATH = './models/classifier.json'
CLASSIFICATION_MODEL_WEIGHTS_PATH = './models/classifier.hdf5'


# image acquisition
def rsize(img):
    if img.shape != (450, 600, 3):
        # REF: https://stackoverflow.com/a/48121983/10309266
        img = cv2.resize(img, dsize=(600, 450),
                         interpolation=cv2.INTER_CUBIC)
        return img

def read(path):
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img:
            return rsize(img)
        else:
            return 'Oops!'
    else:
        return FileNotFoundError


def procedure(img):
    # segmentation
    mask, img = otsuThreshold(img)

    # feature extraction
    crc, ira, irb, irc, ird, avgRadius, c = GeometricFeatures(mask)
    c_bb, c_bg, c_br, c_gg, c_br, c_gr, c_rr, adhocb1, adhocg1, adhocr1, adhocb2, adhocg2, adhocr2 = ColorFeatures(
        mask, img, avgRadius, c)
    Bmean, Gmean, Rmean, Bstd, Gstd, Rstd, Bsk, Gsk, Rsk = TextureFeatures(
        mask, img)

    # classification
    with open(CLASSIFICATION_MODEL_ARCH_PATH) as json_file:
        model = keras.models.model_from_json(json_file)
    model.load_weights(CLASSIFICATION_MODEL_WEIGHTS_PATH)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    score = model.evaluate(X, Y, verbose=0)

    return score


def main_app(path):
    img = read(path)
    return procedure(img)