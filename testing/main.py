#Credits:- https://github.com/zhixuhao/unet
from model import *
from data import *
import skimage.io as io
import cv2
import numpy as np

# imports - third party imports
from skimage import morphology
from skimage.filters import threshold_otsu

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

#Unused
def segment(img):
    """
    input: img is numpy.ndarray
    return: numpy.ndarry
    """

    img = img.astype(np.uint8)


    global_thresh = threshold_otsu(img)
    binary_global = img < global_thresh

    # temp = morphology.remove_small_objects(
    #     binary_global, min_size=500, connectivity=1)
    # mask = morphology.remove_small_holes(temp, 500, connectivity=2)

    mask = binary_global

    for i in range(len(img)):
        for j in range(len(img[0])):
            if mask[i][j] == False:
                mask[i][j] = (0, 0, 0)
            else:     
                mask[i][j] = (255, 255, 255)

    return mask.astype(np.uint64)


def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #img = segment(img)
        
        #Write issues present; Results in black image
        #cv2.imwrite(os.path.join(save_path,"%d_predict.png"%i),img)

        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)



data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

#trainer
#Give path to where you've kept the data
#Second and third is name of folder for data, mask images respectively
myGene = trainGenerator(2,'data/','img','mask',data_gen_args,save_to_dir = None,)


#Init the model
model = unet()

#Retraining
#model = Model.load_weights()

#Incase of system crash, save to model
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=2,callbacks=[model_checkpoint])


# serialize model to JSON
model_json = model.to_json()
with open("TAKEDIS.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("WEIGHT.h5")
print("Saved model to disk")

#Perform testing for evaluation
testGene = testGenerator("data/test")
results = model.predict_generator(testGene,10,verbose=1)     
saveResult("data/test/results",results)