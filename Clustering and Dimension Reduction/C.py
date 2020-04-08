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


data1=pd.read_csv('AllBooks_baseline_DTM_Labelled.csv') #importing the data
n=len(data1.axes[0]) #to find number of rows in each attribute

data1.drop([13], inplace = True) # removing "Buddhism_Ch14" from Dataframe

data1.reset_index(drop=True, inplace=True) # to adjust indices accordingly

data1["Unnamed: 0"]=data1["Unnamed: 0"].str.replace(r'_Ch', '') # to remove "_Ch" 
data1["Unnamed: 0"] = data1["Unnamed: 0"].str.replace('\d+', '') # to remove number after book name
print(data1.iloc[0:20,0:5])

data2=data1.drop(data1.columns[0],axis='columns')
data=data2.iloc[:,:].values
data=np.array(data)  # tranforming into numpy array
data=np.float64(data)   
num_rows=data.shape[0]
num_columns=data.shape[1]
for j in range(0,num_columns):  
    count=0
    for i in range(0,num_rows):
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
# Data can be used for furthur calculations
print("\n \n print final value of data: ")
print(data[0:15,0:15])


# In[2]:


def distance(centroids,data,K):
    distance1=np.ones(K)
    for i in range(K):
        distance1[i]=np.dot(centroids[i],data) # centroid was not normalise so we need to normlised it.
        dinominator=np.sqrt(np.dot(centroids[i],centroids[i])*np.dot(data,data)) # data was already normalised so np.dot(data,data)=1.
        distance1[i]=distance1[i]/dinominator
    return(np.exp(-distance1))


# In[3]:


K=8  # to obtain K=8 clusters of documents
iterations=300
num_rows1=data.shape[0]
num_columns1=data.shape[1]

centroids=np.random.rand(K,num_columns1)


# In[12]:



belongs_to=np.ones(len(data))
SSE=np.zeros(iterations)

for itr in range(iterations):
    for i in range(len(data)):
        distances=distance(centroids,data[i],K)

        SSE[itr]+=(distances.sum())

        min_dist_index=0
        min_dist=distances[0]
        for j in range(1,K):
            if(distances[j]<min_dist):
                min_dist=distances[j]
                min_dist_index=j

        belongs_to[i]=min_dist_index


    centroids[:]=0
    count=np.zeros(K)
    for i in range(len(belongs_to)):
        centroids[int(belongs_to[i])]+=data[i]
        count[int(belongs_to[i])]+=1

    for i in range(K):
        if(count[i]!=0):
            centroids[i]/=count[i]

        
    cluster=[[],[],[],[],[],[],[],[]]
    for i in range(len(belongs_to)):
        cluster[int(belongs_to[i])].append(i)
        
    sorted_cluster=sorted(cluster)
        
#return(sorted_cluster,centroids,SSE)


# In[13]:


Output=sorted_cluster
print(Output)


# In[6]:


#import matplotlib.pyplot as plt
plt.figure(figsize=[15,8])
plt.plot(range(iterations),SSE,c='g')
plt.xlabel('Iteration')
plt.ylabel('SSE')
plt.show()


# In[7]:


f=open("E:/SEMESTER 6/ML-CS60050/Assgnment/Third/kmeans.txt",'w')
for i in range(8):
    for j in sorted_cluster[i]:
        f.write(str(j))
        f.write(",")
    f.write("\n")
f.close()


# In[8]:


plt.scatter(data[:,0],data[:,1],c='blue',label='unclustered data')
#plt.xlabel('Rows')
#plt.ylabel('columns')
plt.legend()
plt.title('Plot of data points')
plt.show()


# In[9]:


#print(np.dot(data[1],data[1]))


# In[10]:


#f.write


# In[11]:


print(len(sorted_cluster))


# In[ ]:




