#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D
)


# In[2]:


fashion_mnist=keras.datasets.fashion_mnist


# In[3]:


(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()


# In[4]:


train_images=train_images/255.0
test_images=test_images/255.0


# In[5]:


train_images=train_images.reshape(len(train_images),28,28,1)
test_images=test_images.reshape(len(test_images),28,28,1)


# In[6]:




INPUT_SHAPE = (28,28,1)
NUM_CLASSES = 10

model = keras.Sequential()
model.add(
    Conv2D(
        filters=16,
        kernel_size=3,
        activation='relu',
        input_shape=INPUT_SHAPE
    )
)
model.add(Conv2D(16, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(rate=0.25))
model.add(Conv2D(32, 3, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(NUM_CLASSES, activation='softmax'))


# In[7]:


model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# In[8]:


mode_fit=model.fit(train_images,train_labels, epochs=5,
    verbose=1)


# In[18]:


get_acc = mode_fit.history
accuracy = get_acc['accuracy'][1] 
accuracy


# In[19]:


f= open("cnn_acc.txt","w+")
f.write(str(round(accuracy*100,2)))
f.close()
print("Accuracy of model is = " , round(accuracy*100,2) ,"%")


# In[ ]:




