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
n=len(data1.axes[0]) #to find number of rows in each attribute

data1.drop([13], inplace = True) # removing "Buddhism_Ch14" from Dataframe

data1.reset_index(drop=True, inplace=True) # to adjust indices accordingly

data1["Unnamed: 0"]=data1["Unnamed: 0"].str.replace(r'_Ch', '') # to remove "_Ch" 
data1["Unnamed: 0"] = data1["Unnamed: 0"].str.replace('\d+', '') # to remove number after book name
#print(data1.iloc[0:20,0:])

data_f=data1.drop(data1.columns[0],axis='columns')
#data=data2.iloc[:,:].values
'''data=np.array(data)  # tranforming into numpy array
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
print(data[0:15,0:15])'''

print(data_f.head())


# In[2]:


# Importing standardscalar module  
from sklearn.preprocessing import StandardScaler 
  
scalar = StandardScaler() 
  
# fitting 
scalar.fit(data_f) 
scaled_data = scalar.transform(data_f) 
  
# Importing PCA 
from sklearn.decomposition import PCA 
  
# Let's say, components = 2 
pca = PCA(n_components = 100) 
pca.fit(scaled_data) 
x_pca = pca.transform(scaled_data) 
  
#x_pca.shape 
df=x_pca # for k- means clustering
data_agglo=x_pca # for agglomorative clustering
# giving a larger plot

plt.figure(figsize =(8, 6)) 
  
plt.scatter(df[:, 0], df[:, 1]) 
  


# In[3]:


num_rows=df.shape[0]
num_columns=df.shape[1]
print(num_columns)


# In[4]:


# components 
print(pca.components_ )


# In[5]:


print(x_pca[0:15,:])
#df=x_pca #now df is final array to reduced features


# In[6]:


#Applying K-mean used method on this to find 8 clusters

data=np.array(df)  # tranforming into numpy array
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


f=open("E:/SEMESTER 6/ML-CS60050/Assgnment/Third/kmeans_reduced.txt",'w')
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


# Applying heirarical method on data_agglo array




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


data_a=data_agglo

k_value=8 # no. of cluster =8
#distances=[]
distances=[]
for i in range(len(data_a)):
    row=[]
    for j in range(i):
        row.append(cosine_similarity_distance(data_a[i],data_a[j]))
    distances.append(row)
    del(row)


cluster=[]
for i in range(len(data)):
    cluster.append([i])

while(len(cluster)>k_value): 
    min1=10000
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


f=open("E:/SEMESTER 6/ML-CS60050/Assgnment/Third/agglomerative_reduced.txt",'w')
for i in range(len(sorted_cluster)):
    for j in sorted_cluster[i]:
        f.write(str(j))
        f.write(",")
    f.write("\n")
f.close()  
# Actual code ends here






# In[8]:


len(data_a[0])


# In[ ]:




