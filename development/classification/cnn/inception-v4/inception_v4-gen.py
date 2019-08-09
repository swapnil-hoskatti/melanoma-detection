import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten, Input,
						  MaxPooling2D, concatenate)
from keras.layers.convolutional import (AveragePooling2D, MaxPooling2D)
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
Implementation of Inception Network v4 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.
"""

TH_BACKEND_TH_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_th_dim_ordering_th_kernels.h5"
TH_BACKEND_TF_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_tf_dim_ordering_th_kernels.h5"
TF_BACKEND_TF_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_tf_dim_ordering_tf_kernels.h5"
TF_BACKEND_TH_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_th_dim_ordering_tf_kernels.h5"


def conv_block(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1), bias=False):
	if K.image_dim_ordering() == "th":
		channel_axis = 1
	else:
		channel_axis = -1

	x = Conv2D(nb_filter, nb_row, nb_col, subsample=subsample, border_mode=border_mode, bias=bias)(x)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)
	return x


def inception_stem(input):
	if K.image_dim_ordering() == "th":
		channel_axis = 1
	else:
		channel_axis = -1

	# Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
	x = conv_block(input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
	x = conv_block(x, 32, 3, 3, border_mode='valid')
	x = conv_block(x, 64, 3, 3)

	x1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)
	x2 = conv_block(x, 96, 3, 3, subsample=(2, 2), border_mode='valid')

	x = concatenate([x1, x2], axis=channel_axis)

	x1 = conv_block(x, 64, 1, 1)
	x1 = conv_block(x1, 96, 3, 3, border_mode='valid')

	x2 = conv_block(x, 64, 1, 1)
	x2 = conv_block(x2, 64, 1, 7)
	x2 = conv_block(x2, 64, 7, 1)
	x2 = conv_block(x2, 96, 3, 3, border_mode='valid')

	x = concatenate([x1, x2], axis=channel_axis)

	x1 = conv_block(x, 192, 3, 3, subsample=(2, 2), border_mode='valid')
	x2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)

	x = concatenate([x1, x2], axis=channel_axis)
	return x


def inception_A(input):
	if K.image_dim_ordering() == "th":
		channel_axis = 1
	else:
		channel_axis = -1

	a1 = conv_block(input, 96, 1, 1)

	a2 = conv_block(input, 64, 1, 1)
	a2 = conv_block(a2, 96, 3, 3)

	a3 = conv_block(input, 64, 1, 1)
	a3 = conv_block(a3, 96, 3, 3)
	a3 = conv_block(a3, 96, 3, 3)

	a4 = AveragePooling2D()(input)
	a4 = conv_block(a4, 96, 1, 1)

	m = concatenate([a1, a2, a3, a4], axis=channel_axis)
	return m


def inception_B(input):
	if K.image_dim_ordering() == "th":
		channel_axis = 1
	else:
		channel_axis = -1

	b1 = conv_block(input, 384, 1, 1)

	b2 = conv_block(input, 192, 1, 1)
	b2 = conv_block(b2, 224, 1, 7)
	b2 = conv_block(b2, 299, 7, 1)

	b3 = conv_block(input, 192, 1, 1)
	b3 = conv_block(b3, 192, 7, 1)
	b3 = conv_block(b3, 224, 1, 7)
	b3 = conv_block(b3, 224, 7, 1)
	b3 = conv_block(b3, 299, 1, 7)

	b4 = AveragePooling2D()(input)
	b4 = conv_block(b4, 128, 1, 1)

	m = concatenate([b1, b2, b3, b4], axis=channel_axis)
	return m


def inception_C(input):
	if K.image_dim_ordering() == "th":
		channel_axis = 1
	else:
		channel_axis = -1

	c1 = conv_block(input, 299, 1, 1)

	c2 = conv_block(input, 384, 1, 1)
	c2_1 = conv_block(c2, 299, 1, 3)
	c2_2 = conv_block(c2, 299, 3, 1)
	c2 = concatenate([c2_1, c2_2], axis=channel_axis)

	c3 = conv_block(input, 384, 1, 1)
	c3 = conv_block(c3, 448, 3, 1)
	c3 = conv_block(c3, 512, 1, 3)
	c3_1 = conv_block(c3, 299, 1, 3)
	c3_2 = conv_block(c3, 299, 3, 1)
	c3 = concatenate([c3_1, c3_2], axis=channel_axis)

	c4 = AveragePooling2D()(input)
	c4 = conv_block(c4, 299, 1, 1)

	m = concatenate([c1, c2, c3, c4], axis=channel_axis)
	return m


def reduction_A(input):
	if K.image_dim_ordering() == "th":
		channel_axis = 1
	else:
		channel_axis = -1

	r1 = conv_block(input, 384, 3, 3, subsample=(2, 2), border_mode='valid')

	r2 = conv_block(input, 192, 1, 1)
	r2 = conv_block(r2, 224, 3, 3)
	r2 = conv_block(r2, 299, 3, 3, subsample=(2, 2), border_mode='valid')

	r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

	m = concatenate([r1, r2, r3], axis=channel_axis)
	return m


def reduction_B(input):
	if K.image_dim_ordering() == "th":
		channel_axis = 1
	else:
		channel_axis = -1

	r1 = conv_block(input, 192, 1, 1)
	r1 = conv_block(r1, 192, 3, 3, subsample=(2, 2), border_mode='valid')

	r2 = conv_block(input, 299, 1, 1)
	r2 = conv_block(r2, 299, 1, 7)
	r2 = conv_block(r2, 320, 7, 1)
	r2 = conv_block(r2, 320, 3, 3, subsample=(2, 2), border_mode='valid')

	r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

	m = concatenate([r1, r2, r3], axis=channel_axis)
	return m


def create_inception_v4(nb_classes=7, load_weights=False):
	'''
	Creates a inception v4 network

	:param nb_classes: number of classes.txt
	:return: Keras Model with 1 input and 1 output
	'''

	if K.image_dim_ordering() == 'th':
		init = Input((3, 299, 299))
	else:
		init = Input((299, 299, 3))

	# Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
	x = inception_stem(init)

	# 4 x Inception A
	for i in range(4):
		x = inception_A(x)

	# Reduction A
	x = reduction_A(x)

	# 7 x Inception B
	for i in range(7):
		x = inception_B(x)

	# Reduction B
	x = reduction_B(x)

	# 3 x Inception C
	for i in range(3):
		x = inception_C(x)

	# Average Pooling
	x = AveragePooling2D()(x)

	# Dropout
	x = Dropout(0.8)(x)
	x = Flatten()(x)

	# Output
	out = Dense(output_dim=nb_classes, activation='softmax')(x)

	model = Model(init, out, name='Inception-v4')

	return model



if __name__ == '__main__':
	K.clear_session()
	filepath = 'inception-v4-model-SGD'

	model = create_inception_v4()

	sgd = SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)
	K.set_value(sgd.lr, 0.5 * K.get_value(sgd.lr))

	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=sgd,
				  metrics=['accuracy'])

	checkpoint_path = filepath + ".hdf5"
	checkpoint = ModelCheckpoint(
		checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

	callbacks_list = [checkpoint]

	datagen = ImageDataGenerator(
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest'
	)

	train_data = datagen.flow_from_directory(
		directory='./all/train/',
		target_size=(299, 299),
		batch_size=10
	)

	validate_data = datagen.flow_from_directory(
		directory='./all/validate/',
		target_size=(299, 299),
		batch_size=10
	)

	test_data = datagen.flow_from_directory(
		directory='./all/test/',
		target_size=(299, 299),
		batch_size=10
	)

	STEP_SIZE_TRAIN = train_data.n//train_data.batch_size
	STEP_SIZE_VALID = validate_data.n//validate_data.batch_size
	STEP_SIZE_PREDICT = test_data.n//test_data.batch_size

	# K.set_value(model.optimizer.lr, 0.001)
	model.fit_generator(
		generator=train_data,
		validation_data=validate_data,
		steps_per_epoch=STEP_SIZE_TRAIN,
		validation_steps=STEP_SIZE_VALID,
		epochs=200,
		verbose=True,
		workers=10,
		callbacks=callbacks_list,
		use_multiprocessing=True,
	)

	validation = model.evaluate_generator(
		generator=validate_data,
		steps=STEP_SIZE_VALID
	)

	prediction = model.predict_generator(
		generator=test_data,
		steps=STEP_SIZE_PREDICT
	)

	print("Evaluation from validation data:\n" + validation)
	print("Evaluation from predicted data:\n" + prediction)
