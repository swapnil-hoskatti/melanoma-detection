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
from . import os, cv2, io, keras, np, morphology, cnn_dict

# imports - standard imports
import multiprocessing
import pickle

# imports - third party randoms
import sklearn
from sklearn.externals import joblib


# constants - global constants for classification
DNN_CLASSIFICATION_MODEL_ARCH_PATH = "../core/models/deep-classify.json"
DNN_CLASSIFICATION_MODEL_WEIGHTS_PATH = "../core/models/deep-classify.h5"

CNN_CLASSIFICATION_MODEL_ARCH_PATH = "../core/models/vgg_16-resized.json"
CNN_CLASSIFICATION_MODEL_WEIGHTS_PATH = "../core/models/vgg_16-model-val_acc-82-acc-72.hdf5"

NEW_CNN_CLASSIFICATION_MODEL_ARCH_PATH = "../core/models/vgg_16-og.json"
NEW_CNN_CLASSIFICATION_MODEL_WEIGHTS_PATH = "../core/models/cnn-model-epoch_01-acc_0.00-valacc_0.90.hdf5"

LOG_CLASSIFICATION_MODEL = "../core/models/logistic_model.pkl"
NORMALIZER = "../core/models/scaler.pkl"

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
	return
	"""
	[Deprecated] dont use
	description: main algorithm
	params: np.ndarray -> uint8 :: values = (0, 255)
	returns: int (label data) :: values = (0, 6)
	"""
	# pre processing
	hair_rem = remove_hair(img)

	# segmentation
	unet_mask = cv2.cvtColor(unetSegment(hair_rem), cv2.COLOR_GRAY2BGR)
	unet_mask = unet_mask.astype(np.uint8)

	otsu_mask = otsuThreshold(hair_rem) * 255

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

	# mask.dtype -> bool
	mask = mainBlob(np.array(temp, dtype=np.uint8))

	roi = getROI(img, np.array(mask))

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
		+ [
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

	# DNN here
	# loaded_model_json = open(DNN_CLASSIFICATION_MODEL_ARCH_PATH).read()
	# model = keras.models.model_from_json(loaded_model_json)
	# model.load_weights(DNN_CLASSIFICATION_MODEL_WEIGHTS_PATH)
	# model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
	# score = int(model.predict([features])[0][0])
	# print(f"Stage 3: Prediction Done ~ {score}")

	# CNN here
	json_file = open(CNN_CLASSIFICATION_MODEL_ARCH_PATH).read()
	model_cnn = keras.models.model_from_json(json_file)
	model_cnn.load_weights(CNN_CLASSIFICATION_MODEL_WEIGHTS_PATH)
	model_cnn.compile(
		loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
	)
	score = int(
		model_cnn.predict_classes(
			[
				cv2.resize(
					roi, dsize=(256, 256), interpolation=cv2.INTER_CUBIC
				).reshape((1, 256, 256, 3))
			]
		)
	)
	print(f"Stage 3: CNN Prediction Done ~ {score}")
	# 5 - VASC ??? everything gets predicted as 5. needs more testing
	print("---" * 10)

	return score


def parallel_procedure(img):
	manager = multiprocessing.Manager()
	masks = manager.dict()
	features_dict = manager.dict()

	# function definitions
	def unet(hair_rem):
		unet_mask = cv2.cvtColor(unetSegment(hair_rem), cv2.COLOR_GRAY2BGR)
		unet_mask = unet_mask.astype(np.uint8)
		masks["unet"] = unet_mask

	def otsu(hair_rem):
		otsu_mask = otsuThreshold(hair_rem) * 255
		masks["otsu"] = otsu_mask

	def combine_masks(unet_mask, otsu_mask):
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
		return np.array(temp, dtype=np.uint8)

	def geo_color(mask, img):
		crc, ira, irb, irc, ird, avgRadius, c = GeometricFeatures(mask)
		c_bb, c_bg, c_br, c_gg, c_gr, c_rr, adhocb1, adhocg1, adhocr1, adhocb2, adhocg2, adhocr2 = ColorFeatures(
			mask, img, avgRadius, c
		)
		features_dict["geo_color"] = [
			crc,
			ira,
			irb,
			irc,
			ird,

			c_gr,
			c_br,
			c_bg,

			c_rr,
			c_gg,
			c_bb,

			adhocb1,
			adhocg1,
			adhocr1,
			adhocr2,
			adhocb2,
			adhocg2,
		]

	def texture(mask, img):
		texture_features = list(TextureFeatures(mask, img))
		features_dict["texture"] = texture_features

	# pre processing
	hair_rem = remove_hair(img)

	# segmentation
	process_unet = multiprocessing.Process(target=unet, args=(hair_rem,))
	process_otsu = multiprocessing.Process(target=otsu, args=(hair_rem,))
	process_otsu.start()
	process_unet.start()
	process_otsu.join()
	process_unet.join()

	while True:
		if len(masks) is 2:
			final_mask = combine_masks(masks["unet"], masks["otsu"])
			break

	mask = mainBlob(final_mask)

	# error: divide by zero
	from core.geometricFeature import GeometricFeatures

	try:
		crc, ira, irb, irc, ird, avgradius, c = GeometricFeatures(mask)
	except ZeroDivisionError:
		mask = final_mask
		try:
			crc, ira, irb, irc, ird, avgradius, c = GeometricFeatures(mask)
		except ZeroDivisionError:
			mask = masks["otsu"]

	# roi for visualization + CNN
	roi = getROI(img, np.array(mask))
	print("Stage 1: Segmentation Done")

	# feature extraction
	process_bins = multiprocessing.Process(target=texture, args=(mask, img))
	process_bins.start()
	process_bins.join()

	process_col_geo = multiprocessing.Process(target=geo_color, args=(mask, img))
	process_col_geo.start()
	process_col_geo.join()

	while True:
		if len(features_dict) is 2:
			print("Stage 2: Feature Extraction Done")
			features = np.array([[
				round(x, ROUND_FACTOR)
				for x in features_dict["geo_color"] + features_dict["texture"]
			]])
			break

	# Convert NaN to 0
	loc_vector = np.isnan(features)
	features[loc_vector] = 0

	# Normalizer
	scaler = joblib.load(NORMALIZER)

	try:
		features_norm = scaler.transform(features)
	except Exception as e:
		print(e)
		features_norm = features

	# Logistic Regressor
	model_lr = joblib.load(LOG_CLASSIFICATION_MODEL)
	pred_logistic = model_lr.predict(features_norm)[0]
	print(f"Stage 3.1: LR Prediction Done ~ {pred_logistic}")

	# CNN: VGG-16
	json_file = open(NEW_CNN_CLASSIFICATION_MODEL_ARCH_PATH).read()
	model_cnn = keras.models.model_from_json(json_file)
	model_cnn.load_weights(NEW_CNN_CLASSIFICATION_MODEL_WEIGHTS_PATH)
	model_cnn.compile(
		loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
	)
	pred_cnn = cnn_dict[
		int(
			model_cnn.predict_classes(
				[
					cv2.resize(
						roi, dsize=(450, 600), interpolation=cv2.INTER_CUBIC
					).reshape((1, 450, 600, 3))
				]
			)
		)
	]
	print(f"Stage 3.2: CNN Prediction Done ~ {pred_cnn}")
	print("---" * 10)

	io.imsave("static/hair_rem.jpg", hair_rem)
	io.imsave("static/unet_mask.jpg", np.array(masks["unet"], dtype=np.uint8) * 255)
	io.imsave("static/otsu_mask.jpg", masks["otsu"])
	io.imsave("static/combined_mask.jpg", np.array(final_mask, dtype=np.uint8))
	cv2.imwrite("static/final_mask.jpg", np.array(mask, dtype=np.uint8).astype(np.uint8) * 255)
	io.imsave("static/roi.jpg", roi)

	# remove data from dicts after each run
	features_dict.clear()
	masks.clear()

	return (features, features_norm, pred_logistic, pred_cnn)


def main_app(path):
	"""
	description: importable to FLASK app
	params: str (path)
	"""
	img = read(path)
	io.imsave("static/img.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	return parallel_procedure(img)
