#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import pandas as pd
from pprint import pprint
import math
import time
import sys
import random
import copy
import os
from numpy import linalg as np_la
import warnings
warnings.filterwarnings("ignore")


data=pd.read_csv('AllBooks_baseline_DTM_Labelled.csv') #importing the data
#n=len(data.axes[0]) #to find number of rows in each attribute

#data.drop([13], inplace = True) # removing "Buddhism_Ch14" from Dataframe
data=data.drop([13], axis=0)

data.reset_index(drop=True, inplace=True) # to adjust indices accordingly

data["Unnamed: 0"]=data["Unnamed: 0"].str.replace(r'_Ch', '') # to remove "_Ch" 
data["Unnamed: 0"] = data["Unnamed: 0"].str.replace('\d+', '') # to remove number after book name


# In[2]:


#data["Unnamed: 0"]


# In[3]:



# H(Y) - Entropy of class labels
def Entropy_of_class_labels(classes,data):
    
    HY=0
    for i in range(len(classes)):
        if(len(classes[i])!=0):
            p=len(classes[i])/len(data)
            HY=HY+(p*np.log2(p))

    HY=-HY
    return HY


# In[4]:


# H(C) - Entropy of cluster labels
def Entropy_of_cluster_labels(clusters,data):
    
    H_C=0
    for i in range(len(clusters)):
        if(len(clusters[i])!=0):
            p=len(clusters[i])/len(data)
            H_C=H_C+(p*np.log2(p))

    H_C=-H_C
    return H_C


# In[5]:


# Calculation of H(Y|C)=conditional entropy of class labels for clustering C

def Conditional_Entropy(clusters,classes):
    #start
    H_Y_C=0

    for i in range(len(clusters)):
        count=0
        for j in range(len(clusters[i])):
            if(clusters[i][j] in classes[i]):
                count=count+1
        if(count!=0):
            p=count/len(clusters[i])
            H_Y_C=H_Y_C+p*np.log2(p)

        H_Y_C=H_Y_C/len(clusters)

    H_Y_C=-H_Y_C
    return H_Y_C


# In[7]:


#Main code is here
data_labels=np.array(data['Unnamed: 0'])   

temp=0;
for item in data['Unnamed: 0']:
    data_labels[temp]=item
    temp=temp+1
# class labels for 'Unnamed: 0' defined below
class_labels=['Buddhism','TaoTeChing','Upanishad','YogaSutra','BookOfProverb', 'BookOfEcclesiastes', 'BookOfEccleasiasticus','BookOfWisdom']
for i in range(len(data_labels)):
    for j in range(len(class_labels)):
        #print("data_labels:",data_labels)
        #print("class_labels",class_labels)
        if(data_labels[i]==class_labels[j]):
            data_labels[i]=j
        
#print(data_labels)
data=data.drop(['Unnamed: 0'], axis=1) #replace the class labels with the new one (data_labels)
data.insert(0, 'Unnamed: 0', data_labels)


classes=[[],[],[],[],[],[],[],[]]     # stores document number in each class
document_labels=np.array(data['Unnamed: 0'])


for i in range(len(document_labels)):
        classes[document_labels[i]].append(i)


# In[8]:




# for "agglomerative.txt"


file = open("E:/SEMESTER 6/ML-CS60050/Assgnment/Third/agglomerative.txt","r")
clusters=[[],[],[],[],[],[],[],[]]
index=0

for line in file:
  
    #Let's split the line into an array called "fields" using the "," as a separator:
    fields = line.split(",")
    
    for i in range(len(fields)):
        clusters[index].append(fields[i])
        
    index=index+1
    
file.close()


for i in range(len(clusters)):
    if(len(clusters[i])!=0):
        item=clusters[i][-1]
        clusters[i][-1]=item[0:len(item)-1]
    

    

## H(Y) - Entropy of class labels

HY=Entropy_of_class_labels(classes,data)


## H(C) - Entropy of cluster labels

HC=Entropy_of_cluster_labels(clusters,data)


# Calculation of H(Y|C)

HYC=Conditional_Entropy(clusters,classes)


IYC=HY-HYC


## NMI(Normalized Mutual Information) = (2*I(Y;C))/(H(Y)+H(C))

NMI=(2*IYC)/(HY+HC)

print("NMI score of agglomerative.txt : ",NMI)


# In[9]:




# for "kmeans.txt"

file = open("E:/SEMESTER 6/ML-CS60050/Assgnment/Third/kmeans.txt","r")
clusters=[[],[],[],[],[],[],[],[]]
index=0

for line in file:
  
    #Let's split the line into an array called "fields" using the "," as a separator:
    fields = line.split(",")
    
    for i in range(len(fields)):
        clusters[index].append(fields[i])
        
    index=index+1
    
file.close()


for i in range(len(clusters)):
    if(len(clusters[i])!=0):
        item=clusters[i][-1]
        clusters[i][-1]=item[0:len(item)-1]
    

    

## H(Y) - Entropy of class labels

HY=Entropy_of_class_labels(classes,data)

## H(C) - Entropy of cluster labels

HC=Entropy_of_cluster_labels(clusters,data)

# Calculation of H(Y|C)

HYC=Conditional_Entropy(clusters,classes)


IYC=HY-HYC


## NMI(Normalized Mutual Information) = (2*I(Y;C))/(H(Y)+H(C))

NMI=(2*IYC)/(HY+HC)

print("NMI score of kmeans.txt : ",NMI)


# In[10]:




# for "kmeans.txt"

file = open("E:/SEMESTER 6/ML-CS60050/Assgnment/Third/kmeans_reduced.txt","r")
clusters=[[],[],[],[],[],[],[],[]]
index=0

for line in file:
  
    #Let's split the line into an array called "fields" using the "," as a separator:
    fields = line.split(",")
    
    for i in range(len(fields)):
        clusters[index].append(fields[i])
        
    index=index+1
    
file.close()


for i in range(len(clusters)):
    if(len(clusters[i])!=0):
        item=clusters[i][-1]
        clusters[i][-1]=item[0:len(item)-1]
    

    

## H(Y) - Entropy of class labels

HY=Entropy_of_class_labels(classes,data)

## H(C) - Entropy of cluster labels

HC=Entropy_of_cluster_labels(clusters,data)

# Calculation of H(Y|C)

HYC=Conditional_Entropy(clusters,classes)


IYC=HY-HYC


## NMI(Normalized Mutual Information) = (2*I(Y;C))/(H(Y)+H(C))

NMI=(2*IYC)/(HY+HC)

print("NMI score of kmeans_reduced.txt : ",NMI)


# In[11]:




# for "agglomerative_reduced.txt"

file = open("E:/SEMESTER 6/ML-CS60050/Assgnment/Third/agglomerative_reduced.txt","r")
clusters=[[],[],[],[],[],[],[],[]]
index=0

for line in file:
  
    #Let's split the line into an array called "fields" using the "," as a separator:
    fields = line.split(",")
    
    for i in range(len(fields)):
        clusters[index].append(fields[i])
        
    index=index+1
    
file.close()


for i in range(len(clusters)):
    if(len(clusters[i])!=0):
        item=clusters[i][-1]
        clusters[i][-1]=item[0:len(item)-1]
    

    

## H(Y) - Entropy of class labels

HY=Entropy_of_class_labels(classes,data)


## H(C) - Entropy of cluster labels

HC=Entropy_of_cluster_labels(clusters,data)


# for H(Y|C)

HYC=Conditional_Entropy(clusters,classes)


IYC=HY-HYC


NMI=(2*IYC)/(HY+HC)

print("NMI score of agglomerative_reduced.txt : ",NMI)


# In[12]:


'''def NMI(file, data):
    
 #   file = open("E:/SEMESTER 6/ML-CS60050/Assgnment/Third/agglomerati","r")
    clusters=[[],[],[],[],[],[],[],[]]
    index=0

    for line in file:
  
        #Let's split the line into an array called "fields" using the "," as a separator:
        fields = line.split(",")
    
        for i in range(len(fields)):
            clusters[index].append(fields[i])
        
        index=index+1
    
    file.close()


    for i in range(len(clusters)):
        if(len(clusters[i])!=0):
            item=clusters[i][-1]
            clusters[i][-1]=item[0:len(item)-1]
    

    

    ## H(Y) - Entropy of class labels

    HY=0
    for i in range(len(classes)):
        if(len(classes[i])!=0):
            p=len(classes[i])/len(data)
            HY=HY+(p*np.log2(p))

    HY=-HY


    ## H(C) - Entropy of cluster labels

    HC=0
    for i in range(len(clusters)):
        if(len(clusters[i])!=0):
            p=len(clusters[i])/len(data)
            HC=HC+(p*np.log2(p))

    HC=-HC

    # Calculation of H(Y|C)

    HYC=0

    for i in range(len(clusters)):
        count=0
        for j in range(len(clusters[i])):
            if(clusters[i][j] in classes[i]):
                count=count+1
        if(count!=0):
            p=count/len(clusters[i])
            HYC=HYC+p*np.log2(p)

        HYC=HYC/len(clusters)

    HYC=-HYC

    IYC=HY-HYC


## NMI(Normalized Mutual Information) = (2*I(Y;C))/(H(Y)+H(C))

    NMI=(2*IYC)/(HY+HC)
    return NMI
#print("Normalized Mutual Information (NMI) for agglomerative.txt : ",NMI)'''


# In[ ]:




