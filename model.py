from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from configuration import config

import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class CNN_model():

    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same', input_shape=config['cnn_input_shape']))
        model.add(Conv2D(32, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu',
                  kernel_initializer='he_uniform'))
        model.add(Dense(4, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def resnet_model(self):
        resnet_50_model = Sequential()
        pretrained_model = tf.keras.applications.ResNet50(
            include_top=False, input_shape=config['cnn_input_shape'], pooling='avg', classes=4, weights='imagenet')
        for layer in pretrained_model.layers:
            layer.trainable = False

        resnet_50_model.add(pretrained_model)
        resnet_50_model.add(Flatten())
        resnet_50_model.add(Dense(512, activation='relu'))
        resnet_50_model.add(Dense(4, activation='softmax'))
        resnet_50_model.summary()
        resnet_50_model.compile(optimizer=Adam(
            learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        # learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return resnet_50_model

    def train_model(self, model, train_X, train_y, val_X, val_y, epochs, batch_size,  early_stopping_patience, logfile_name):
        callbacks = []
        callbacks.append(CSVLogger(logfile_name, append=True))
        callbacks.append(ModelCheckpoint(
            'Models4/model-{epoch:03d}.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto'))
        callbacks.append(EarlyStopping(verbose=1,
                         patience=early_stopping_patience))
        model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                  callbacks=callbacks, validation_data=[val_X, val_y])

    def evaluate_model(self, model, test_X, tesy_y):
        score = model.evaluate(test_X, tesy_y, verbose=1)
        print('\nKeras CNN #1C - accuracy:', score[1], '\n')

    def plot_CM(self, model, test_X, test_y):
        pred = model.predict(test_X)
        pred_classes = np.argmax(pred, axis=1)

        cm = confusion_matrix(test_y, pred_classes)
        print(cm)
        df_cm = pd.DataFrame(cm, index=[i for i in "ELMN"],
                             columns=[i for i in "ELMN"])
        sn.heatmap(df_cm, annot=True, annot_kws={
                   "size": 10}, fmt='g')  # size: font size  fmt:'g' means not to show as scientific notation.

        plt.savefig('./confusion_matrix.png')
        plt.show()
