#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
os.chdir('D:/python using jupyter/Data Preprocessing')


# In[2]:


sfpermits = pd.read_csv('Building_Permits.csv')


# In[3]:


sfpermits.head()


# In[5]:


np.random.seed(0)


# # What percent of data is missing 

# In[6]:


totalCells = np.product(sfpermits.shape) # to calculate total no of shells in the dataframe

missingCount = sfpermits.isnull().sum() # to count no of missing values per column

totalMissing = missingCount.sum() # to count the total missing

print('The sfpermits dataset contains', round(((totalMissing/totalCells)*100),2),"%","missing values.") # calculate percentage of missing values


# In[8]:


sfpermits.isnull().sum()


# In[9]:


missingCount[['Street Number Suffix', 'Zipcode']]


# In[11]:


print("Percent missing data in Street Number Suffix column =", (round(((missingCount['Street Number Suffix'] / sfpermits.shape[0]) * 100), 2)))
print("Percent missing data in Zipcode column =", (round(((missingCount['Zipcode'] / sfpermits.shape[0]) * 100), 2)))


# In[12]:


len(sfpermits)


# In[13]:


missingCount/len(sfpermits) # to know the percentage of missing values in each column


# In[14]:


# now remove all the columns which contains empty values
sfpermitscleanCols = sfpermits.dropna(axis=1)
sfpermitscleanCols.head()


# In[23]:


print('Columns in the original dataset: %d \n' % sfpermits.shape[1])
print('Columns in the original dataset: %d \n' % sfpermitscleanCols.shape[1])


# with the above procedure we missed a lot of things hence its not sugges

# # Try replacing all the NaN's in the sf_permit data with the one that comes directly after it and then [replace all the reamining na's with 0]

# In[25]:


imputesfpermits = sfpermits.fillna(method='ffill',axis=0).fillna('0') 
imputesfpermits.head()


# In[ ]:




