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




data1=pd.read_csv('AllBooks_baseline_DTM_Labelled.csv') #importing the data
n1=len(data1.axes[0]) #to find number of rows in each attribute

data1.drop([13], inplace = True) # removing "Buddhism_Ch14" from Dataframe

data1.reset_index(drop=True, inplace=True) # to adjust indices accordingly

data1["Unnamed: 0"]=data1["Unnamed: 0"].str.replace(r'_Ch', '') # to remove "_Ch" 
data1["Unnamed: 0"] = data1["Unnamed: 0"].str.replace('\d+', '') # to remove number after book name
print(data1.iloc[0:20,:])

data2=data1.drop(data1.columns[0],axis='columns') # removing string column
print('final data to be used : \n',data2.iloc[0:20,0:10])


# In[2]:


def cosine_similarity_distance(point_1, point_2): # to find distance
    distance=0.0
    distance=np.dot(point_1, point_2)
    dinominator=np.sqrt(np.dot(point_1,point_1)*np.dot(point_2,point_2))
    distance=distance/dinominator
    
    return(np.exp(-distance))
    
    

def add_points(point_1, point_2):
    for i in range(0, len(point_1)):
        point_1[i] = float(point_1[i]) + float(point_2[i])
    return point_1


# In[3]:


data=data2.values

k_value=8 # no. of cluster =8
#distances=[]
distances=[]
for i in range(len(data)):
    row=[]
    for j in range(i):
        row.append(cosine_similarity_distance(data[i],data[j]))
    distances.append(row)
    del(row)


cluster=[]
for i in range(len(data)):
    cluster.append([i])

while(len(cluster)>k_value): 
    min1=1000
    combine=[0,1]
    for i in range(len(cluster)):
        for j in range(i+1,len(cluster)):
            #temp=single_linkage(cluster,distances,i,j)
            min_value=1000
            for m in cluster[i]:
                for n in cluster[j]:
                    if(m>n):
                        if(min_value>distances[m][n]):
                            min_value=distances[m][n]
                        else:
                            min_value=distances[n][m]
            temp=min_value  
        
            
            if(min1>temp):
                min1=temp
                combine[0]=i
                combine[1]=j
    cluster[combine[0]]=cluster[combine[0]]+cluster[combine[1]]
    del(cluster[combine[1]])
       
sorted_cluster=sorted(cluster)


# In[4]:


#k_value=8 # 8 cluster we taking here
#agglomerative_local(data2, k_value)
#cluster=centroid_points
#for i in range(len(cluster)):
#    cluster[i]=sorted(cluster[i]) # sorting each row of final cluster or say centroid point

#sorted_cluster=sorted(cluster) # sorting the final value
print('cluster size: ',len(sorted_cluster))
j=1
for i in sorted_cluster: # total 8 cluster
    print("#")
    #print("cluster-",j,"=",i)
    #print("\n \n")
    #j=j+1
for i in sorted_cluster: # total 8 cluster
    print("#")
    print("cluster-",j,"=",i)
    print("\n \n")
    j=j+1    


# In[5]:


f=open("E:/SEMESTER 6/ML-CS60050/Assgnment/Third/agglomerative.txt",'w')
for i in range(len(sorted_cluster)):
    for j in sorted_cluster[i]:
        f.write(str(j))
        f.write(",")
    f.write("\n")
f.close()  
# Actual code ends here


# In[6]:


#this part not neccessary so I'am using libaray for it because it has not been asked in question
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
Z = linkage(data2, 'single')
fig = plt.figure(figsize=(25, 25))
dn = dendrogram(Z)
#plt.hlines(y=190,xmin=0,xmax=2000,lw=10,linestyles='--')
#plt.text(x=900,y=36,s='Horizontal line crossing 8 vertical lines',fontsize=20)
#plt.grid(True)


# In[ ]:




