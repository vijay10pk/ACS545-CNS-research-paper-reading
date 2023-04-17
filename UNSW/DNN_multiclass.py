#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger


# In[17]:


traindata = pd.read_csv('traindata.csv', index_col = 0)
testdata = pd.read_csv('testdata.csv', index_col = 0)


# In[18]:


traindata.head()


# In[20]:


X = traindata.iloc[:,0:42]
Y = traindata.iloc[:,42]
C = testdata.iloc[:,42]
T = testdata.iloc[:,0:42]


# In[25]:


Y.unique()


# In[22]:


trainX = np.array(X)
testT = np.array(T)

trainX.astype(float)
testT.astype(float)

scaler = Normalizer().fit(trainX)
trainX = scaler.transform(trainX)

scaler = Normalizer().fit(testT)
testT = scaler.transform(testT)

y_train1 = np.array(Y)
y_test1 = np.array(C)

y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)

X_train = np.array(trainX)
X_test = np.array(testT)


# In[23]:


batch_size = 64


# In[24]:


print("1 Layer DNN")


# In[14]:


y_train


# In[29]:


model = Sequential()
model.add(Dense(1024,input_dim=42,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(10))
model.add(Activation('softmax'))


# In[30]:


# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/dnn1layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('kddresults/dnn1layer/training_set_dnnanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, validation_data=(X_test, y_test),batch_size=batch_size, epochs=100, callbacks=[checkpointer,csv_logger])
model.save("kddresults/dnn1layer/dnn1layer_model.hdf5")


# In[ ]:


print("2 Layer DNN")


# In[ ]:


model = Sequential()
model.add(Dense(1024,input_dim=42,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(768,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(10))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/dnn2layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('kddresults/dnn2layer/training_set_dnnanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, validation_data=(X_test, y_test),batch_size=batch_size, epochs=100, callbacks=[checkpointer,csv_logger])
model.save("kddresults/dnn2layer/dnn2layer_model.hdf5")


# In[ ]:


print("3 Layer DNN")


# In[ ]:


model = Sequential()
model.add(Dense(1024,input_dim=42,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(768,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(512,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(10))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/dnn3layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('kddresults/dnn3layer/training_set_dnnanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=100, callbacks=[checkpointer,csv_logger])
model.save("kddresults/dnn3layer/dnn3layer_model.hdf5")


# In[ ]:


print("4 Layer DNN")


# In[ ]:


model = Sequential()
model.add(Dense(1024,input_dim=42,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(768,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(512,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(256,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(10))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/dnn4layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('kddresults/dnn4layer/training_set_dnnanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, validation_data=(X_test, y_test),batch_size=batch_size, epochs=100, callbacks=[checkpointer,csv_logger])
model.save("kddresults/dnn4layer/dnn4layer_model.hdf5")


# In[ ]:


print("5 Layer DNN")


# In[ ]:


model = Sequential()
model.add(Dense(1024,input_dim=42,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(768,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(512,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(256,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(128,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(10))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/dnn5layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('kddresults/dnn5layer/training_set_dnnanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, validation_data=(X_test, y_test),batch_size=batch_size, epochs=100, callbacks=[checkpointer,csv_logger])
model.save("kddresults/dnn5layer/dnn5layer_model.hdf5")

