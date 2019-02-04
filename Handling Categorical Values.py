#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.DataFrame([['female','New York','low',4],['female','London','medium',3],['male','New Delhi','high',2]],
                   columns = ['Gender','City','Temprature','Rating'])


# In[3]:


data


# # Types of Encoding

# 1. Numeric Encoding

# In[4]:


from sklearn.preprocessing import LabelEncoder
data['City_encoded'] = LabelEncoder().fit_transform(data['City'])
data


# 2. Specifying an encoder

# In[5]:


data['Temprature_encoded'] = data['Temprature'].map({'low':0,'medium':1,'high':2})
data


# 3. Binary encoding

# In[8]:


data['Male'] = data['Gender'].map( {'male':1, 'female':0} )
data[['Gender', 'Male']]


# 4. One hot encoding

# In[9]:


pd.get_dummies(data['City'],prefix='City')


# To concatenate these new feautres with the existing data, do the following:
# 

# In[10]:


data = pd.concat([data,pd.get_dummies(data['City'],prefix='City')],axis=1)
data[['City','City_London','City_New Delhi','City_New York']]


# In[ ]:




