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


# def predict(self, X, Y, weight_1, weight_2):
#         ypred = np.array([np.argmax(self._forward_pass(x_, Y, weight_1, weight_2)) for x_ in X], dtype=np.int)
#         return ypred

def _dotprod(self, a, b): # for dot product
        return sum([a_ * b_ for (a_, b_) in zip(a, b)])
    
    
def _update_weights(x, eta):
        for i, layer in enumerate(self.network):
            # Grab input values
            if i == 0: inputs = x
            else: inputs = [node_['output'] for node_ in self.network[i-1]]
            # Update weights
            for node in layer:
                for j, input in enumerate(inputs):
                    # dw = - learning_rate * (error * transfer') * input
                    node['weights'][j] += - eta * node['delta'] * input    

    


# In[5]:


def backward(X, Y, x_end, x_middle_1, x_middle_2,  w1, w2, w3):
    learning_rate = 0.01
    if(Y==1):
        Y = [1,0,0]
    elif(Y==2):
        Y = [0,1,0]
    else:
        Y = [0,0,1]

    #Lost Function
    # cost = 0;
    # for i in range(3):
    # 	cost += Y[i] * math.log(x_end[i],2)
    # print -(cost/3)

    derivation_cost_function = np.zeros(3)
    for i in range(3):
        derivation_cost_function[i] = x_end[i] - Y[i]


    
    # for finding relu derivative
    '''def diff_relu(x):
    for i in range(len(x)):
        if(x[i]<=0):
            x[i] = 0
        else:
            x[i] = 1

    return x'''
    x1=x_middle_2
    for i in range(len(x1)):
        if(x1[i]<=0):
            x1[i] = 0
        else:
            x1[i] = 1
    x_middle_2_deri = x1 #x_middle_2_deri = diff_relu(x_middle_2)
    delta_1 = np.zeros(32) #deltas are numbered in the reverse direction
    for i in range(32):
        delta_1[i] = x_middle_2_deri[i] * np.dot(w3[i], derivation_cost_function)

    delta_2 = np.zeros(64)
    # relu derivative
    x2=x_middle_1
    for i in range(len(x2)):
        if(x2[i]<=0):
            x2[i] = 0
        else:
            x2[i] = 1
    x_middle_1_deri = x2 #x_middle_1_deri = diff_relu(x_middle_1)

    for i in range(32):
        delta_2[i] = x_middle_1_deri[i] * np.dot(w2[i], delta_1)


    #Now update weights
    for i in range(32):
        for j in range(3):
            w3[i,j] = w3[i,j] - learning_rate * derivation_cost_function[j] * x_middle_2[i]

    for i in range(64):
        for j in range(32):
            w2[i,j] =w2[i,j] - learning_rate * delta_1[j] * x_middle_1[i]


    for i in range(7):
        for j in range(64):
            w1[i,j] = w1[i,j] - learning_rate * X[i] * delta_2[j] 

    return w1 , w2 , w3


# In[6]:


def forward(X,Y, w1, w2, w3):
    learning_rate = 0.01
    n = len(X)
    #print("1")
    for input in range(n):
        x_middle_1 = np.zeros(64)
        x_middle_2 =np.zeros(32)
        for i in range(64):
            #applying relu function
            '''def relu(x):
                if(x<0):
                    return 0
                else:
                    return x'''

            
            relu1=np.dot(X[input],w1[:,i]) + 1 #1 for bias
            if(relu1<0):
                x_middle_1[i] =0
            else:
                x_middle_1[i] =relu1

            #x_middle_1[i] = relu(np.dot(X[input],weight_1[:,i]) + 1) 

        for i in range(32):
            #applying relu function
            relu2=np.dot(x_middle_1,w2[:,i]) + 1 #1 for bias
            if(relu2<0):
                x_middle_2[i] =0
            else:
                x_middle_2[i] =relu2
            #x_middle_2[i] = relu(np.dot(x_middle_1,weight_2[:,i]) + 1) #1 for bias
        #print("1")
        x_end = np.zeros(3)
        for i in range(3):
            #applying relu function
            relu3=np.dot(x_middle_2,w3[:,i]) + 1 #1 for bias
            if(relu3<0):
                x_end[i] =0
            else:
                x_end[i] =relu3
            #x_end[i] = relu(np.dot(x_middle_2,weight_3[:,i]) + 1) #1 for bias

        #Applying Softmax to final layer x_end
        '''def softmax(x):
        deno = sum(np.exp(val) for val in x)
        for i in range(len(x)):
            x[i] = np.exp(x[i]) / deno
        return x'''
        deno = sum(np.exp(val) for val in x_end)
        for i in range(len(x_end)):
            x_end[i] = np.exp(x_end[i]) / deno  #x_end = softmax(x_end)


        

        #backpropagation will return the updated weights
        w1 , w2 , w3 = backward( X[input], Y[input], x_end, x_middle_1, x_middle_2, w1, w2, w3)

    return w1, w2, w3


# In[7]:


def testing(w1, w2, w3, X, Y):
    n = len(X)
    count = 0;
    acc = np.zeros(n)
    for input in range(n):
        x_middle_1 = np.zeros(64)
        x_middle_2 =np.zeros(32)
        for i in range(64):
            #applying relu function
            '''def relu(x):
                if(x<0):
                    return 0
                else:
                    return x'''


            relu1=np.dot(X[input],w1[:,i]) + 1 #1 for bias
            if(relu1<0):
                x_middle_1[i] =0
            else:
                x_middle_1[i] =relu1

            #x_middle_1[i] = relu(np.dot(X[input],weight_1[:,i]) + 1) 

        for i in range(32):
            #applying relu function
            relu2=np.dot(x_middle_1,w2[:,i]) + 1 #1 for bias
            if(relu2<0):
                x_middle_2[i] =0
            else:
                x_middle_2[i] =relu2
            #x_middle_2[i] = relu(np.dot(x_middle_1,weight_2[:,i]) + 1) #1 for bias

        x_end = np.zeros(3)
        for i in range(3):
            #applying relu function
            relu3=np.dot(x_middle_2,w3[:,i]) + 1 #1 for bias
            if(relu3<0):
                x_end[i] =0
            else:
                x_end[i] =relu3
            #x_end[i] = relu(np.dot(x_middle_2,weight_3[:,i]) + 1) #1 for bias

        #Applying Softmax to final layer x_end
        #Applying Softmax to final layer x_end
        '''def softmax(x):
        deno = sum(np.exp(val) for val in x)
        for i in range(len(x)):
            x[i] = np.exp(x[i]) / deno
        return x'''
        deno = sum(np.exp(val) for val in x_end)
        for i in range(len(x_end)):
            x_end[i] = np.exp(x_end[i]) / deno  #x_end = softmax(x_end)
        if(Y[input]==1):
            y = [1,0,0]
        elif(Y[input]==2):
            y = [0,1,0]
        else:
            y = [0,0,1]

        if(x_end[1] > x_end[2]):
            if(x_end[0] < x_end[1]):
                x = [0,1,0]
            else:
                x = [1,0,0]
        else:
            if(x_end[0] < x_end[2]):
                x = [0,0,1]
            else:
                x = [1,0,0]

        temp = 0
        for i in range(3):
            if(x[i] != y[i]):
                temp = 1;

        if(temp==0):
            count += 1;
    result_f=count * 100 / n
    return result_f


# In[8]:


acc_train = []
acc_test = []
for batch in range(num_rows_train//32):

    data_batch = np.array(train_x[32*(batch):32*(batch+1),:])
    output_batch = np.array(train_y[32*(batch):32*(batch + 1)])

    #Initialise weight matrix
    weight_1 = np.zeros(shape=(7,64))
    weight_2 = np.zeros(shape=(64,32))
    weight_3 = np.zeros(shape=(32,3))
    for i in range(7):
        for j in range(64):
            weight_1[i,j] = random.uniform(-1,1)

    for i in range(64):
        for j in range(32):
            weight_2[i,j] = random.uniform(-1,1)

    for i in range(32):
        for j in range(3):
            weight_3[i,j] = random.uniform(-1,1)


    for epochs in range(1,201):
        #learning_rate = 0.01

        #Training
        weight_1, weight_2, weight_3 = forward(data_batch, output_batch, weight_1, weight_2, weight_3)

        #Testing
        accuracy_train = testing(weight_1, weight_2, weight_3, data_batch, output_batch)
        accuracy_test = testing(weight_1, weight_2, weight_3, test_x, test_y)

        if(epochs%10 == 0):
            acc_train.append(accuracy_train)
            acc_test.append(accuracy_test)
    print("#")


# In[9]:


acc_train = np.array(acc_train)
acc_test = np.array(acc_test)
print("Final training accuracy: ",acc_train[-1])
print("Final testing accuracy: ",acc_test[-1])

xaxis = np.linspace(0,5,100)

plt.figure()
#plt.subplot(121,200,500)


plt.plot(xaxis,acc_train,label='Train')
plt.plot(xaxis,acc_test,label='Test')
plt.xlabel('Input data')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




