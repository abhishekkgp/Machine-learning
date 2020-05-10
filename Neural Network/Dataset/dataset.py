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


# In[2]:


data = [] #actual data

with open("E:/SEMESTER 6/ML-CS60050/Assgnment/Fourth/data.txt") as f: #taking input
    for line in csv.reader(f, dialect="excel-tab"): #reading the data
        data.append(line)



data = np.array(data)  

num_rows = data.shape[0]   #len(data)
num_colm = data.shape[1]  #len(data[:][0])

data=np.float64(data)
num_colm = num_colm - 1


# In[3]:


#z-normalisation
for i in range(num_colm): 
    x = np.array(data[:,i])
    mean = np.mean(x)  # mean value
    std = np.std(x)   # standard deviation

    for j in range(num_rows):
        data[j,i] = (data[j,i] - mean)/std


# In[4]:


#taking 80% random data for training
no_of_row_train = int(num_rows*0.8)
indices_value = np.random.choice(data.shape[0], no_of_row_train, replace=False)
data_train = data[indices_value]


#print ()
print ("Training Data: ", data_train[0:5,:])


#taking 20% random data for testing
indices_negative_val = []
for i in range(num_rows):
	indices_negative_val.append(i)
indices_negative_val = list(set(indices_negative_val) - set(indices_value))
data_test = np.array(data[indices_negative_val])

#print ("Testing Data")
print ("Testing Data: ",data_test[0:5,:])


# In[5]:


#Saving inside a csv file seperately
np.savetxt('C:/Users/Abhishek/Desktop/a4/training_data.csv', data_train, delimiter=',')
np.savetxt('C:/Users/Abhishek/Desktop/a4/testing_data.csv', data_test, delimiter=',')


# In[6]:


print("number of rows in training data: ",no_of_row_train)


# In[7]:


print("number of rows in actural data:",num_rows)


# In[8]:


print("number of rows in testing data:",num_rows-no_of_row_train)


# In[ ]:




