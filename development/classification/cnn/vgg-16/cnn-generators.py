import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


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
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=7, activation='softmax'))

    return model


if __name__ == '__main__':
    K.clear_session()
    filepath = 'vgg_16-model-SGD'

    model = vgg_16()
    
    sgd = SGD()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd,
                  metrics=['accuracy'])

    checkpoint_path = filepath + ".{epoch:02d}-{val_acc:.2f}-{loss:.2f}.hdf5"

    checkpoint_val_acc = ModelCheckpoint(
        checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint_loss = ModelCheckpoint(
        checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='max')
    
    callbacks_list = [checkpoint_val_acc, checkpoint_loss]

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

    test_data = datagen.flow_from_directory(
        directory='./all/test/',
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

    save_model(model)

    print("Evaluation from validation data:\n" + validation)
    print("Evaluation from predicted data:\n" + prediction)

    
