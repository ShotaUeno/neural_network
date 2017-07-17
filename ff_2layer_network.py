import sys, os
from mnist import load_mnist  #datasetディレクトリ内のmnist.pyから、load_mnist関数をインポート
import numpy as np
import matplotlib.pylab as plt
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0) # オーバーフロー対策
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy(x, t):
    delta = 1e-7
    if x.ndim == 1:
        t = t.reshape(1, t.size)
        x = x.reshape(1, y.size)

    batch_size = x.shape[0]
    return -np.sum(t*np.log(x))/batch_size


def slope(f, x):
    h = 1e-4
    result = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        buf = x[i]

        x[i] = float(buf) + h
        fh_plus = f(x)
        
        x[i] = float(buf) - h
        fh_minus = f(x)

        result[i] = (fh_plus - fh_minus)/(2*h)
        x[i] = buf
        it.iternext()
    return result

class Network:
    def __init__(self, input_layer, hidden_layer, output_layer):
        self.parameter = {}
        self.parameter['w1'] = 0.01*np.random.randn(input_layer, hidden_layer) #784x100個の乱数
        self.parameter['b1'] = np.zeros(hidden_layer)
        self.parameter['w2'] = 0.01*np.random.randn(hidden_layer, output_layer) #100x10個の乱数
        self.parameter['b2'] = np.zeros(output_layer)

    def predict(self, x):
        w1 = self.parameter['w1']
        b1 = self.parameter['b1']
        w2 = self.parameter['w2']
        b2 = self.parameter['b2']
        a1 = np.dot(x,w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(a1,w2) + b2
        y = softmax(a2)
        return y

    def loss_func(self, x, t):
        y = self.predict(x)
        return cross_entropy(y, t)

    def ff_learning(self, x, t):
        loss_w = lambda w: self.loss_func(x, t) #変数として関数に代入するための定義
        updt = {}
        updt['w1'] = slope(loss_w, self.parameter['w1'])
        updt['b1'] = slope(loss_w, self.parameter['b1'])
        updt['w2'] = slope(loss_w, self.parameter['w2'])
        updt['b2'] = slope(loss_w, self.parameter['b2'])
        return updt


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_size = x_train.shape[0]
batch_size = 3
train_loss_list = []

network = Network(784,50,10)

for_n = 50
start = time.time()
for i in range(for_n):
    mini_batch = np.random.choice(train_size, batch_size)  #0~60000のうちbatch_size個の乱数を計算
    x_batch = x_train[mini_batch]
    t_batch = t_train[mini_batch]

    grads = network.learning(x_batch,t_batch)

    for j in ('w1', 'b1', 'w2', 'b2'):
        network.parameter[j] -= 0.1*grads[j]

#for i in range(for_n)のスコープここまで

end = time.time()
print("学習時間: " + str(end - start))
