import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Embedding,\
    LSTM, Bidirectional, Lambda, Concatenate, Add
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import prepare
import model as build_model

mxlen = 32
embedding_dim = 50
lstm_unit = 128
MLP_unit = 128
epochs = 100
batch_size = 64

#train_json = 'nlvr\\train\\train.json'
#train_img_folder = 'nlvr\\train\\images'

def main(dataset="babl"):
    if dataset == "nlvr":
        data = prepare.load_data(train_json)
        data = prepare.tokenize_data(data, mxlen)
        imgs, ws, labels = prepare.load_images(train_img_folder, data, debug=True)
        data.clear()
        imgs_mean = np.mean(imgs)
        imgs_std = np.std(imgs - imgs_mean)
        imgs = (imgs - imgs_mean) / imgs_std

    else:
        inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test = prepare.get_babl_data()


    epochs = 100
    model = build_model.model()
    model.fit([inputs_train, queries_train], answers_train, validation_split=0.1, epochs=epochs)
    model.save('model')





