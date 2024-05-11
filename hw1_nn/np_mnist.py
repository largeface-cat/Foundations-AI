# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""
import time

# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy')  # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy')  # (60000, )
y_train = np.eye(10)[y_train]  # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy')  # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy')  # (10000,)
y_val = np.eye(10)[y_val]  # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy')  # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy')  # (10000,)
y_test = np.eye(10)[y_test]  # (10000, 10), one-hot编码


# 定义激活函数
def relu(x):
    '''
    relu函数
    '''
    return np.maximum(x, 0)


def relu_prime(x):
    '''
    relu函数的导数
    '''
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def leaky_relu_prime(x):
    return np.where(x > 0, 1, 0.01)


# 输出层激活函数
def f(x):
    '''
    softmax函数, 防止除0
    '''
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))


def f_prime(x):
    '''
    softmax函数的导数
    '''
    return f(x) * (1 - f(x))


# 定义损失函数
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    return -np.sum(y_true * np.log(y_pred))


def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    return -y_true / y_pred


def cross_entropy_loss_softmax(y_true, a):
    '''
    dLoss/dz = a - y
    '''
    return a - y_true


# 定义权重初始化函数
def init_weights(shape=()):
    '''
    初始化权重
    '''
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0 / shape[0]), size=shape)


# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''

    def __init__(self, input_size, hidden_size, hidden_size2, output_size, lr=0.01):
        '''
        初始化网络结构
        '''
        self.W1 = init_weights((input_size, hidden_size))  # 输入层到隐藏层的权重矩阵
        self.b1 = init_weights((1, hidden_size))
        self.W2 = init_weights((hidden_size, hidden_size2))
        self.b2 = init_weights((1, hidden_size2))
        self.W3 = init_weights((hidden_size2, output_size))
        self.b3 = init_weights((1, output_size))

        self.lr = lr

    def forward(self, x):
        '''
        前向传播
        '''
        z1 = np.matmul(x, self.W1) + self.b1
        a1 = leaky_relu(z1)
        z2 = np.matmul(a1, self.W2) + self.b2
        a2 = tanh(z2)
        z3 = np.matmul(a2, self.W3) + self.b3
        a3 = f(z3)
        return a3

    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''
        batch_size = 0
        batch_loss = 0
        batch_acc = 0
        self.grads_W3 = np.zeros_like(self.W3)
        self.grads_b3 = np.zeros_like(self.b3)
        self.grads_W2 = np.zeros_like(self.W2)
        self.grads_b2 = np.zeros_like(self.b2)
        self.grads_W1 = np.zeros_like(self.W1)
        self.grads_b1 = np.zeros_like(self.b1)
        for x, y in zip(x_batch, y_batch):
            # 前向传播
            z1 = np.matmul(x, self.W1) + self.b1
            a1 = leaky_relu(z1)
            z2 = np.matmul(a1, self.W2) + self.b2
            a2 = tanh(z2)
            z3 = np.matmul(a2, self.W3) + self.b3
            a3 = f(z3)
            # 计算损失和准确率
            loss = loss_fn(y, a3)
            acc = (np.argmax(y) == np.argmax(a3))

            # 反向传播

            delta_L = cross_entropy_loss_softmax(y, a3)

            # backward
            self.grads_W3 += np.matmul(a2.T, delta_L)
            self.grads_b3 += delta_L

            delta_2 = np.matmul(delta_L, self.W3.T) * tanh_prime(z2)
            self.grads_W2 += np.matmul(a1.T, delta_2)
            self.grads_b2 += delta_2

            delta_1 = np.matmul(delta_2, self.W2.T) * leaky_relu_prime(z1)
            self.grads_W1 += np.outer(x, delta_1)
            self.grads_b1 += delta_1

            # 更新权重

            batch_size += 1
            batch_loss += loss
            batch_acc += acc
        self.grads_W3 /= batch_size
        self.grads_b3 /= batch_size
        self.grads_W2 /= batch_size
        self.grads_b2 /= batch_size
        self.grads_W1 /= batch_size
        self.grads_b1 /= batch_size
        self.W3 -= self.lr * self.grads_W3
        self.b3 -= self.lr * self.grads_b3
        self.W2 -= self.lr * self.grads_W2
        self.b2 -= self.lr * self.grads_b2
        self.W1 -= self.lr * self.grads_W1
        self.b1 -= self.lr * self.grads_b1
        batch_loss /= batch_size
        batch_acc /= batch_size

        return batch_loss, batch_acc


if __name__ == '__main__':
    # 训练网络

    # %%
    net = Network(input_size=784, hidden_size=256, hidden_size2=64, output_size=10, lr=0.04)
    loss_list = []
    acc_list = []
    val_acc_list = []
    for epoch in range(10):
        if epoch > 2:
            lr = 0.01
        elif epoch > 5:
            lr = 0.003
        elif epoch > 7:
            lr = 0.001
        losses = []
        accuracies = []
        p_bar = tqdm(range(0, len(X_train), 64))
        for i in p_bar:
            x_batch = X_train[i:i + 64]
            y_batch = y_train[i:i + 64]
            loss, acc = net.step(x_batch, y_batch)
            a2 = net.forward(x_batch)
            losses.append(loss)
            accuracies.append(acc)
            p_bar.update(1)

        loss_list.append(l := (sum(losses) / len(losses)))
        acc_list.append(a := (sum(accuracies) / len(accuracies)))
        print('epoch:', epoch, 'train loss:', l, 'train acc:', a)

        # 在验证集上验证
        a2 = net.forward(X_val)
        acc = np.mean(np.argmax(a2, axis=1) == np.argmax(y_val, axis=1))
        val_acc_list.append(acc)
        print('val acc:', acc)

    plt.plot(loss_list)
    plt.plot(acc_list)
    plt.plot(val_acc_list)
    plt.legend(['loss', 'acc', 'val_acc'])
    plt.show()

    # # 在测试集上测试（只在最终完成训练和调参后进行）
    # a2 = net.forward(X_test)
    # acc = np.mean(np.argmax(a2, axis=1) == np.argmax(y_test, axis=1))
    # print('test acc:', acc)
