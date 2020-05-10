#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import csv
import math
import time
import random


# In[2]:


#For training
data_train = []
with open("C:/Users/Abhishek/Desktop/a4/training_data.csv") as f1:
    for line in csv.reader(f1):
        data_train.append(line)
        
        
#data_test = np.array(data_test)
num_rows_train = len(data_train)
num_colm_train = len(data_train[:][0])
data_train = np.array(data_train).astype(np.float)

train_y = np.array(data_train[:,num_colm_train -1])
train_x = np.delete(data_train,num_colm_train-1,1) 
num_colm_train = num_colm_train - 1

print(data_train[0:5,:])


# In[3]:


# testing data

data_test = []
with open("C:/Users/Abhishek/Desktop/a4/testing_data.csv") as f2:
    for line in csv.reader(f2):
        data_test.append(line)

#data_test = np.array(data_test)        
num_rows_test = len(data_test)
num_colm_test = len(data_test[:][0])
data_test = np.array(data_test).astype(np.float)


test_y = np.array(data_test[:,num_colm_test -1])
test_x = np.delete(data_test,num_colm_test-1,1) 

num_colm_test = num_colm_test - 1

print(data_test[0:5,:])


# In[4]:



import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier


# For specification of part1A
mlp1 = MLPClassifier(hidden_layer_sizes=(32),activation="logistic",solver='sgd', batch_size=32,learning_rate="constant",learning_rate_init=0.01,max_iter=200,random_state=10)
mlp1.fit(train_x, train_y)

print("fot part1-A: ")
print("Training set score: %f" % mlp1 .score(train_x, train_y))
print("Test set score: %f" % mlp1 .score(test_x, test_y))


#mlp.classes_


# In[6]:


# For specification of part1B

mlp2 = MLPClassifier(hidden_layer_sizes=(64,32),activation="relu",solver='sgd',batch_size=32,learning_rate="constant",learning_rate_init=0.01,max_iter=200)
mlp2.fit(train_x, train_y)

print("fot part1-B: ")
print("Training set score: %f" % mlp2 .score(train_x, train_y))
print("Test set score: %f" % mlp2 .score(test_x, test_y))


#mlp.classes_


# In[ ]:




