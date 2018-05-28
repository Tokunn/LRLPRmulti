#!/usr/bin/env python2

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, Chain

import sys, time
sys.path.append('../common/loadimg/')
import loadimg

def convorder(x):
    x_tmp = []
    for i in range(x.shape[0]):
        x_tmp.append(x[i].transpose((2, 1, 0)))
    x = np.asarray(x_tmp)
    return x


#train, test = chainer.datasets.get_mnist(ndim=3)
x_train, y_train, x_test, y_test, class_count = loadimg.loadimg()
x_train = convorder(x_train)
x_test = convorder(x_test)

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
                conv1 = L.Convolution2D(3, 20, 5),
                conv2 = L.Convolution2D(20, 50, 5),
                fc1 = L.Linear(800, 500),
                fc2 = L.Linear(500, class_count),
        )
    def __call__(self, x):
        cv1 = self.conv1(x)
        relu = F.relu(cv1)
        h = F.max_pooling_2d(relu, 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.dropout(F.relu(self.fc1(h)), train=(x_train, y_train))
        return self.fc2(h)


model = L.Classifier(Model())
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)

#batchsize = 1000
batchsize = 1

def conv(batch, batchsize):
    x = []
    t = []
    for j in range(batchsize):
        x.append(batch[j][0])
        t.append(batch[j][1])
    return Variable(np.array(x)), Variable(np.array(t))

starttime = time.time()
for n in range(400):
    #for i in chainer.iterators.SerialIterator(train, batchsize, repeat=False):
    #    x, t = conv(i, batchsize)

    #    model.zerograds()
    #    loss = model(x, t)
    #    loss.backward()
    #    optimizer.update()

    x = Variable(x_train)
    t = Variable(y_train)
    model.zerograds()
    loss = model(x, t)
    loss.backward()
    optimizer.update()

    #i = chainer.iterators.SerialIterator(test, batchsize).next()
    #x, t = conv(i, batchsize)
    x = Variable(x_test)
    t = Variable(y_test)
    loss = model(x, t)
    print n, loss.data

print(starttime - time.time())
