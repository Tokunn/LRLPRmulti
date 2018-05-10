#!/usr/bin/env python2

import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, list_pictures, img_to_array
from keras.utils import np_utils
import matplotlib.pyplot as plt

COMMONDIR = '../common/make_image'
TRAINDIR = os.path.join(COMMONDIR, 'train')
TESTDIR = os.path.join(COMMONDIR, 'test')

IMGSIZE = 64
IMGSIZE = 28

def loadimg_one(DIRPATH):
    x = []
    y = []
    img_list = os.listdir(DIRPATH)
    img_count = 0

    for number in img_list:
        dirpath = os.path.join(DIRPATH, number)
        for picture in list_pictures(dirpath):
            img = img_to_array(load_img(picture, target_size=(IMGSIZE, IMGSIZE)))
            x.append(img)
            y.append(img_count)
            #print("Load {0} : {1}".format(picture, img_count))
        img_count += 1

    output_count = img_count
    x = np.asarray(x)
    x = x.astype('float32')
    x = x/255.0
    y = np.asarray(y, dtype=np.int32)
    #y = np_utils.to_categorical(y, output_count)

    return x, y, output_count


def loadimg():
    print("########## loadimg ########")

    x_train, y_train, class_count = loadimg_one(TRAINDIR)
    x_test,  y_test,  _  = loadimg_one(TESTDIR)
    #for i in range(0, x_test.shape[0]):
    #    plt.imshow(x_test[i])
    #    plt.show()

    print("########## END of loadimg ########")
    return x_train,  y_train, x_test, y_test, class_count

if __name__ == '__main__':
    loadimg()
