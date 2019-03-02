# imports - data
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# imports - models and stuff
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, LeakyReLU
from keras.callbacks import ModelCheckpoint

# global constants
num_classes = 7
filepath = "weights-improvement-leaky"

# data loading
df = pd.read_csv("labelled-features-refined.csv")
X = np.array(df.iloc[:, 1:-1])
Y_ = np.array(df['class'])
Y = to_categorical(Y_)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=62)

# dnn model
model = Sequential()
model.add(Dense(23, input_dim=23, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1500, activation='relu'))
model.add(Dense(3500, activation='relu'))
model.add(Dense(4500))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(4500))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(3500, activation='relu'))
model.add(Dense(1500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])


checkpoint_path = filepath + "-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train, Y_train,  validation_split=0.33, epochs=10,
          batch_size=100, callbacks=callbacks_list, verbose=True)

scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model_json = model.to_json()
with open(filepath + ".json", "w") as json_file:
    json_file.write(model_json)
