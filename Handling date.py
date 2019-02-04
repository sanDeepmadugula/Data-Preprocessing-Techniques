#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
os.chdir('D:/python using jupyter/Data Preprocessing')


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
import datetime


# In[3]:


landslides = pd.read_csv('catalog.csv')


# In[4]:


landslides.head()


# In[5]:


np.random.seed(0)


# In[6]:


landslides.shape


# In[7]:


print(landslides['date'].head())


# In[9]:


landslides['date'].dtype


# In[11]:


# converting date columns into date time
landslides['date_parsed'] = pd.to_datetime(landslides['date'],format='%m/%d/%y')


# In[12]:


landslides['date_parsed'].head()


# In[13]:


landslides['date_parsed'].dtype


# In[14]:


day_months_landslided = landslides['date_parsed'].dt.day


# In[16]:


day_months_landslided.head()


# In[17]:


# remove na's
day_months_landslided = day_months_landslided.dropna()


# In[18]:


sns.distplot(day_months_landslided,kde=False,bins=60)


# In[ ]:




