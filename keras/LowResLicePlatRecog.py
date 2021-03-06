
# coding: utf-8

# In[ ]:


import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array
from keras.preprocessing.image import list_pictures, load_img, ImageDataGenerator
from keras.applications import imagenet_utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import datetime, time, os

import sys
sys.path.append('../common/loadimg/')
import loadimg

x_train, y_train, x_test, y_test, class_count = loadimg.loadimg()
y_train = np_utils.to_categorical(y_train, class_count)
y_test = np_utils.to_categorical(y_test, class_count)


c = x_train.shape[0]
#for i in range(0, c, 1000):
    #plt.imshow(x_train[i])
    #plt.show()



model = Sequential()

#model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
#model.add(Activation('relu'))
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#
#model.add(Conv2D(32, (3, 3), padding='same'))
#model.add(Activation('relu'))
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(class_count))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])




filename = 'resizeO10'
filename = "./weight/{0}_{1}.hdf5".format(filename, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
if os.path.exists(filename):
    raise ValueError("[!!] File name collision detected !")

print(datetime.datetime.now())
start_time = time.time()
#history = model.fit(x_train, y_train, batch_size=5, epochs=20,
history = model.fit(x_train, y_train, epochs=20,
                    validation_data = (x_test, y_test), verbose=1)
print(datetime.datetime.now())
print(time.time() - start_time)

model.save_weights(filename)




plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.show()




predict_classes = model.predict_classes(x_test, verbose=1)
mg_df = pd.DataFrame({'predict':predict_classes, 'class':np.argmax(y_test, axis=1)})
pd.crosstab(mg_df['class'], mg_df['predict'])




#c = x_test.shape[0]

#y_test_axis1 = np.argmax(y_test, axis=1)

# random output
#for i in range(0, c, 10):
    #plt.imshow(array_to_img(x_test[i]))
    #plt.show()
    #print(number_list[y_test_axis1[i]])
    #print("predict : " + number_list[predict_classes[i]])
    #if (y_test_axis1[i] != predict_classes[i]):
        #print("[!!] Wrong Answer")
        
# Wrong Answer
#w_count = 0
#for i in range(c):
    #if (y_test_axis1[i] != predict_classes[i]):
        #plt.imshow(array_to_img(x_test[i]))
        #plt.show()
        #print(number_list[y_test_axis1[i]])
        #print("predict : " + number_list[predict_classes[i]])
        #print("[!!] Wrong Answer")
        #w_count += 1
#print("Accuracy: {0}/{1}={2}".format(c-w_count, c, (c-w_count)/float(c)))

