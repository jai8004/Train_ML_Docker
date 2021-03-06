#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score


# In[3]:



# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]


# In[4]:



#Create dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)


# In[5]:



## Concatenate the Data Frames

X=pd.concat([X,geography,gender],axis=1)

## Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)


# In[6]:



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[7]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[9]:




# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# In[10]:


def tweak_model(classifier,layers,neurons,epochs,model_tweak_count):
    print("Model Tweak :" ,model_tweak_count)
    print("Layers Added :",layers)
    print("Neurons Per Layer :",neurons)
    print("Epochs This Layer :",epochs)
    print("********************************")    
    for i in range(layers):
        
        classifier.add(Dense(units = neurons, kernel_initializer = 'he_uniform',activation='relu'))
    return classifier   


# In[22]:



neurons = 10
train_acc = 0
epochs = 90
layers = 1 
flag = 0
model_tweak_count=0



while int(train_acc) < 90:
    if flag ==1 :
        classifer = keras.backend.clear_session()
        neurons = neurons+10
        epochs = epochs+5
        layers=layers+3
        model_tweak_count=model_tweak_count+1
        
        
    classifier = Sequential()
    
    classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))
    
    classifier = tweak_model(classifier,layers,neurons,epochs,model_tweak_count)
    classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model_history=classifier.fit(X_train, y_train,validation_split=0.1, batch_size = 10, epochs = epochs,verbose=1)
   
    prev_acaccuracy=train_acc
    m_history=model_history.history
    train_acc=m_history['accuracy'][epochs-1] 
    train_acc=round(train_acc*100,2)
    
    #prev_acaccuracy=accuracy
   # y_pred = classifier.predict(X_test)
   # y_pred = (y_pred > 0.5)    
   # score=accuracy_score(y_pred,y_test)
  
    #accuracy=round(score*100,2)
    print("Previous Accuracy:",prev_acaccuracy)
    print("Current Accuracy :", train_acc)
    print("Accuracy Imporved by:",train_acc-prev_acaccuracy)
    flag=1
    
    


# In[ ]:


classifier.save('tweak_model.h5')

