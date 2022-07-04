import os
import cv2
import glob
import numpy as np
from random import shuffle
from tensorflow.keras.utils import to_categorical
from configuration import config


class data_loader ():
    def __init__(self, img_size):
        self.img_size = img_size

    def convert_data(self, data_path):
        X = []
        y = []
        WBC_classes = config['WBC_classes']
        for image_file in glob.glob(os.path.join(data_path, "*", "*")):
            img = cv2.imread(image_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            X.append(img)
            label = image_file.split('/')[-2]
            if label == WBC_classes[1]:  # NEUTROPHIL
                y.append(0)
            elif label == WBC_classes[2]:  # EOSINOPHIL
                y.append(1)
            elif label == WBC_classes[3]:  # MONOCYTE
                y.append(2)
            elif label == WBC_classes[4]:  # LYMPHOCYTE
                y.append(3)
        return X, y

    def convert_to_array(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        y = to_categorical(y)
        return X, y

    def convert_training_data(self, trainig_data_path, validation_percentage):
        X, y = self.convert_data(trainig_data_path)
        data = list(zip(X, y))
        shuffle(data)
        X, y = zip(*data)
        len_val = int(len(X) * validation_percentage)
        val_X = X[:len_val]
        val_y = y[:len_val]
        train_X = X[len_val:]
        train_y = y[len_val:]
        train_X, train_y = self.convert_to_array(train_X, train_y)
        val_X, val_y = self.convert_to_array(val_X, val_y)
        return train_X, train_y, val_X, val_y

    def convert_test_data(self, test_data_path):
        X, y = self.convert_data(test_data_path)
        test_X, test_y = self.convert_to_array(X, y)
        return test_X, test_y
