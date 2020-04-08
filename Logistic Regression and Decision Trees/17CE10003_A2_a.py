# -*- coding: utf-8 -*-
"""ML_A2_Q!.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fu5YTS7GHMTuT1pRWentgHIFPbTuP8zm
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('winequality-red.csv', sep=';') #importing the data

print(data.head())

x=data.iloc[1,11]


# #feature=data['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'] 
# feature=data.iloc[:,0:-1]
 
# '''xl_scaled=(xl-xl.min())/(xl.max()-xl.min()) #where "ptp" means "max-min"
# xl_scaled'''
# feature.head()

for i in range(11): # except quality feature every column included here
   x=data.iloc[:,i]
   x_minmax_scaled=(x-x.min())/(x.max()-x.min()) # Normalization using min-max scaling
   data.iloc[:,i]=x_minmax_scaled

print(data.head())

#n=len(data.axes[0]) #to find number of rows in each attribute
#print('Number of rows are: ', n)
n=1599
for i in range(n):
  x=data.iloc[i,11]
  print("x",x)
  if(x<=6) :
    x=0
  else :
    x=1
  data.iloc[i,11]=x   
  
  

print(data.iloc[0:10,:])

print('last row value is', data.iloc[-1,:])

"""# For part B"""

data2=pd.read_csv('winequality-red.csv', sep=';') #importing the data
print(data2.head())

n=len(data2.axes[0]) #to find number of rows in each attribute
print('Number of rows are: ', n)
for i in range(n):
  x=data2.iloc[i,-1] #for 'quality' attribute
  
  if(x<5):
    x=0
  elif(x==6 or x==5):
    x=1
  else:
    x=2
  data2.iloc[i,-1]=x


print(data2.head())

print(data2.iloc[1:20,:])

'''for i in range(11): # except quality feature every column included here
   x=data.iloc[:,i]
   x_zscore=(x-x.mean())/x.std() # Normalization using min-max scaling
   data.iloc[:,i]=x_zscore'''
for i in range(11):
     x=data2.iloc[:,i]
     x_zscore=(x-x.mean())/x.std()
     data2.iloc[:,i]=x_zscore

print(data2.iloc[0:20,:])

# for making four BINs
for i in data2:
  if (i=='quality'):
    break
  assign=pd.cut(data2[i],4,labels=['0','1','2','3'],precision=100)
  data2.drop( [i],inplace=True, axis=1 )
  data2.insert(0,i,assign)

print(data2.iloc[0:20,:])

