#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

import math
import time



# In[3]:


data1=pd.read_csv('AllBooks_baseline_DTM_Labelled.csv') #importing the data
n=len(data1.axes[0]) #to find number of rows in each attribute
print('Number of rows are: ', n)
print('Number of Columns are: ', len(data1.axes[1]))


# In[4]:


print(data1.iloc[0:20,:])


# ##  Dataset Preparation:

# In[5]:


data1.drop([13], inplace = True) # removing "Buddhism_Ch14" from Dataframe
#print(data.iloc[0:20,0:5])


# In[6]:


data1.reset_index(drop=True, inplace=True) # to adjust indices accordingly
print(data1.iloc[0:20,0:5])


# In[7]:


#data.iloc[:,0] = data.iloc[:,0].str.replace(r'_Ch$', '')
data1["Unnamed: 0"]=data1["Unnamed: 0"].str.replace(r'_Ch', '') # to remove "_Ch" 
data1["Unnamed: 0"] = data1["Unnamed: 0"].str.replace('\d+', '') # to remove number after book name
print(data1.iloc[0:20,0:5])


# In[8]:


#data=data1.copy()
data2=data1.drop(data1.columns[0],axis='columns')
data=data2.iloc[:,:].values
data=np.array(data)  # tranforming into numpy array
data=np.float64(data)   
num_rows=data.shape[0]
num_columns=data.shape[1]
#num_rows= len(data.axes[0])           #data.shape[0]
#num_columns= len(data.axes[1])          #data.shape[1]
for j in range(0,num_columns):  #Calculating the tf idf score for each data point using the frequency 
    count=0
    for i in range(0,num_rows):
        #if data[i][j]>0 :
            #count+=data[i][j]>0
    	count+=data[i][j]
    for i in range(0,num_rows):
        data[i][j]=1.0*data[i][j]*math.log(1.0*(1+num_rows)/(1+count))
for i in range(0,num_rows):   #Normalizing each datapoint by dividing by the magnitude
	magnitude=0
	for j in range(0,num_columns):
		magnitude+=data[i][j]*data[i][j]
	magnitude=math.sqrt(magnitude)
	if(magnitude==0):  #There is a single point data point with magnitude zero or which contains an empty row
		continue
	for j in range(0,num_columns):
		data[i][j]/=magnitude
#for i in range(0,num_rows):
	#for j in range(0,num_columns):
print(data[0:20,0:15])





