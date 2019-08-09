import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


def save_model(model):
	model_json = model.to_json()
	with open(filepath + ".json", "w") as json_file:
		json_file.write(model_json)


def vgg_16():
	model = Sequential()

	model.add(Conv2D(input_shape=(256, 256, 3), filters=64,
					 kernel_size=3, activation='relu'))
	model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(strides=2))

	model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
	model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(strides=2))

	model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
	model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
	model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(strides=2))

	model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
	model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
	model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(strides=2))

	model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
	model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
	model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(strides=2))

	model.add(Flatten())
	model.add(Dense(units=4096, activation='relu'))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=4096, activation='relu'))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=7, activation='softmax'))

	return model


if __name__ == '__main__':
	K.clear_session()
	filepath = 'cnn-model'

	model = vgg_16()

	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer='adam',
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
		batch_size=10
	)

	validate_data = datagen.flow_from_directory(
		directory='./all/validate/',
		batch_size=10
	)

	# test_data = datagen.flow_from_directory(
	#     directory='./all/test/',
	# )

	STEP_SIZE_TRAIN = train_data.n//train_data.batch_size
	STEP_SIZE_VALID = validate_data.n//validate_data.batch_size

	model.fit_generator(
		generator=train_data,
		validation_data=validate_data,
		steps_per_epoch=STEP_SIZE_TRAIN,
		validation_steps=STEP_SIZE_VALID,
		epochs=10,
		verbose=True,
		workers=1,
		callbacks=callbacks_list,
		use_multiprocessing=False,
	)

	model.evaluate_generator(generator=validate_data)

	save_model(model)
