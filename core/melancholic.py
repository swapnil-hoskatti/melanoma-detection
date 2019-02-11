#################################################################################
#                       Main Application - Base System                          #
#   image-acquisition -> segmentation -> Feature Extraction -> Classification   #
#################################################################################

# imports - imports for segmentation
from core.segmentation import otsuThreshold, unetSegment

# imports - imports for feature extraction
from core.colorFeature import ColorFeatures
from core.textureFeature import TextureFeatures
from core.geometricFeature import GeometricFeatures

# imports - final imports
from . import os, cv2, io, keras, np, morphology

# constants - global constants for classification
CLASSIFICATION_MODEL_ARCH_PATH = '../core/models/deep-classify.json'
CLASSIFICATION_MODEL_WEIGHTS_PATH = '../core/models/deep-classify.h5'


# image acquisition
def rsize(img):
    if img.shape != (450, 600, 3):
        # REF: https://stackoverflow.com/a/48121983/10309266
        return cv2.resize(img, dsize=(600, 450),
                         interpolation=cv2.INTER_CUBIC)
        
    return img


def read(path):
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img.any():
            return rsize(img)
        else:
            return 'Oops!'
    else:
        return FileNotFoundError


def procedure(img):
    # segmentation
    unet_mask = cv2.cvtColor(unetSegment(img), cv2.COLOR_GRAY2BGR)
    io.imsave('../temp_files/unetSegmentOG.jpg', unet_mask)
    unet_mask = unet_mask.astype(np.uint8)

    mask, img = otsuThreshold(img)
    temp = [[[0,0,0] for x in range(0,600)] for y in range(0,450)]
    for i, n in enumerate(mask):
        for j, m in enumerate(n):
            if m:
                temp[i][j] = [255, 255, 255]
    otsu_mask = np.array(temp, dtype=np.uint8)
    io.imsave('../temp_files/otsuSegmentOG.jpg', otsu_mask)

    # combine unet and otsu's mask
    for i in range(450):
        for j in range(600):
            otsu = otsu_mask[i][j]
            unet  = unet_mask[i][j]
            if (any(unet==[1,1,1]) or any(otsu==[255,255,255])):
                temp[i][j] = [255,]*3
            else:
                temp[i][j] = [0,]*3
    io.imsave('../temp_files/combinedSegmentOG.jpg', np.array(temp))

    print('Stage 1: Segmentation Done')

    # feature extraction
    crc, ira, irb, irc, ird, avgRadius, c = GeometricFeatures(mask)
    c_bb, c_bg, c_br, c_gg, c_gr, c_rr, adhocb1, adhocg1, adhocr1, adhocb2, adhocg2, adhocr2 = ColorFeatures(
        mask, img, avgRadius, c)
    Bmean, Gmean, Rmean, Bstd, Gstd, Rstd, Bsk, Gsk, Rsk = TextureFeatures(
        mask, img)
    print('Stage 2: Feature Extraction Done')

    features = np.array([
        Bmean, Gmean, Rmean, Bstd, Gstd, Rstd, crc, ira, irb, irc,
        ird, c_bb, c_bg, c_br, c_gg, c_br, c_gr, c_rr, adhocb1, adhocg1,
        adhocr1, adhocb2, adhocg2, adhocr2
    ]).reshape((1,24))

   
    # classification
    json_file = open(CLASSIFICATION_MODEL_ARCH_PATH)
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)

    model.load_weights(CLASSIFICATION_MODEL_WEIGHTS_PATH)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    score = int(model.predict([features])[0][0])
    print(f'Stage 3: Prediction Done ~ {score}')

    return score


def main_app(path):
    img = read(path)
    print('Stage 0: Acquisition Done')
    return procedure(img)
