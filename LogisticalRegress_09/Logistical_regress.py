#!usr/bin/env python3
# -*- coding:utf-8 -*-
'''
    Machine Learning 第九次作业：
    （1）不借助任何机器学习工具包，手动实现逻辑回归（logistic regress）模型梯度下降算法；

    （2）数据集可以使用以下代码随机生成：

    def random_data():
        np.random.seed(0)
        X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
        Y = [0] * 20 + [1] * 20
        return X, Y
'''
import numpy as np


def random_data():
    np.random.seed(0)
    X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
    Y = [0] * 20 + [1] * 20
    Y = np.array(Y).reshape(-1,1)
    return X, Y

def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s

def sigmoid_derivative(z):
    s = 1 / (1 + np.exp(-z))
    ds = s * (1 - s)
    return ds

def predict(x, w, b):
    return sigmoid(x.dot(w) + b)

def logistic_regress(x, y):
    # w， b
    w = np.random.randn(2,1)
    b = np.random.randn(40,1)

    # hyper parameters
    learning_rate = 1e-3
    iter = 50000
    print('Start training')
    for t in range(iter):
        # Forward pass
        h = x.dot(w) + b
        y_pred = sigmoid(h)
        # Compute and print loss(using square error)
        loss = np.square(y_pred - y).sum()
        if t % 1000 == 999:
            print(t, loss)
        # Backprop to compute the gradients of w and b
        grad_y_pred = 2 * (y_pred - y)
        grad_h = sigmoid_derivative(grad_y_pred)
        grad_w = x.T.dot(grad_h)
        grad_b = grad_h

        # print('x.shape', x.shape)
        # print('y.shape', y.shape)
        # print('h.shape', h.shape)
        # print('y_pred.shape', y_pred.shape)
        # print('grad_y_pred.shape:', grad_y_pred.shape)
        # print('grad_h.shape:', grad_h.shape)
        # print('grad_w.shape', grad_w.shape)
        # print('grad_b.shape', grad_b.shape)
        # exit(0)

        # Update w and b
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    # test
    acc = ( (predict(x,w,b) > 0.5) == y ).astype('float').mean()*100
    print('The accuracy is :{}%'.format(acc))


def main():
    x, y = random_data()
    logistic_regress(x, y)

if __name__ == '__main__':
    main()