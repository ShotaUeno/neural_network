import sys, os
from mnist import load_mnist
import numpy as np
import matplotlib.pylab as plt
import time

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

class affine_layer:
    def __init__(self, w, b):
        self.x = None
        self.w = w
        self.b = b
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout,axis=0) 
        return dx

class relu_layer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        out = x
        self.mask = (x <= 0)
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class soft_and_loss_layer:  #softmaxとlossを一緒にするとbackwardが簡潔になる
    def __init__(self):
        self.y = None
        self.t = None
        self.l = None

    def forward(self, a, t):
        self.t = t
        self.y = softmax(a)
        self.l = cross_entropy(self.y, t)
        return self.l

    def backward(self, dout):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

class Network:
    def __init__(self, input_layer, hidden_layer, output_layer):
        self.para = {}
        self.para['w1'] = 0.01 * np.random.randn(input_layer, hidden_layer) #乱数が入った784x50の行列
        self.para['b1'] = np.zeros(hidden_layer)
        self.para['w2'] = 0.01 * np.random.randn(hidden_layer, output_layer) #乱数が入った50x10の行列
        self.para['b2'] = np.zeros(output_layer)

        self.layers = {}
        self.layers['aff1'] = affine_layer(self.para['w1'], self.para['b1'])
        self.layers['rel1'] = relu_layer()
        self.layers['aff2'] = affine_layer(self.para['w2'], self.para['b2'])
        
        self.sl_layer = soft_and_loss_layer()

    def predict(self, x):
        for i in ('aff1', 'rel1', 'aff2'):
            x = self.layers[i].forward(x)
        return x

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 :
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def loss_func(self, x, t):
        y = self.predict(x)
        return self.sl_layer.forward(y, t)

    def learning(self, x, t):
        self.loss_func(x, t)    #sl_layerのインスタンス変数に値を代入するために実行
        dout = self.sl_layer.backward(1)
        for i in ('aff2', 'rel1', 'aff1'):
            dout = self.layers[i].backward(dout)
        grads = {}
        grads['w1'] = self.layers['aff1'].dw
        grads['b1'] = self.layers['aff1'].db
        grads['w2'] = self.layers['aff2'].dw
        grads['b2'] = self.layers['aff2'].db
        return grads


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

network = Network(784,50,10)


for_n = 10000
start = time.time()
for i in range(for_n):
    mini_batch = np.random.choice(train_size, batch_size)
    x_batch = x_train[mini_batch]
    t_batch = t_train[mini_batch]

    grads = network.learning(x_batch,t_batch)

    for j in ('w1', 'b1', 'w2', 'b2'):
        network.para[j] -= 0.1*grads[j]

    loss = network.loss_func(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train accuracy = " + str(train_acc) + ", test accuracy = " + str(test_acc))

#for i in range(10000)のスコープここまで
end = time.time()
print("学習時間: " + str(end - start))


x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.legend(loc='lower right')
plt.show()

