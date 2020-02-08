#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# author: cherub
# date: 2020-02-08
'''
    A full_connected network for MNIST from keras.datasets
'''
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np

# 一个简单的全连接层神经网络
def full_connected_network():
    # step-1 数据准备
    print('preparing datasets'.center(100, '-'))
    # 1-1，加载数据集
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 查看数据的维度信息
    print('x_train.shape: {}'.format(x_train.shape))
    print('y_train.shaep: {}'.format(y_train.shape))
    print('x_test.shape: {}'.format(x_test.shape))
    print('y_test.shape: {}'.format(y_test.shape))
    # 1-2，归一化
    scaler = StandardScaler()
    # scaler.transform 用于归一化
    x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
    x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

    # step-2 构建模型
    print('building model'.center(100, '-'))
    # 2-1 搭建模型
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation='relu'))
    model.add(keras.layers.Dense(30, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    # 2-2 编译模型
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print('model information: '.center(100, '-'))
    model.summary()

    # step-3 train
    print('training'.center(100, '-'))
    model.fit(x_train_scaled, y_train,
              epochs=30,
              batch_size=128,
              validation_split=0.2)

    # step-4 evaluate
    print('evaluating'.center(100, '-'))
    model.evaluate(x_test_scaled, y_test)


if __name__ == "__main__":
    full_connected_network()


