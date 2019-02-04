#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
os.chdir('D:/python using jupyter/Data Preprocessing')


# In[3]:


df = pd.read_csv('New_weather.csv')


# In[5]:


df


# In[8]:


type(df.day[0])


# In[9]:


# as the day showing in str we need to convert it into timestamp
df = pd.read_csv('New_weather.csv', parse_dates=['day'])
df


# In[10]:


type(df.day[0])


# In[11]:


# setting the index as day
df.set_index('day',inplace=True)


# In[12]:


df


# In[13]:


new_df = df.fillna('0')
new_df


# In[14]:


# to fillna to each column we will use dictionary
new_df = df.fillna({
    'temprature':0,
    'windspeed0':0,
    'event':'no event'
})
new_df


# In[15]:


# in order to fill the forward value
new_df = df.fillna(method='ffill')
new_df


# In[17]:


new_df = df.fillna(method='bfill')
new_df


# In[18]:


new_df = df.fillna(method='bfill',axis='columns') # it will fill columns by backward with columns
new_df


# In[19]:


new_df = df.fillna(method='ffill',limit=1)  # or limit=2 # it will fill forward value to NA but only one value bca limit we have given as 1
new_df


# In[20]:


new_df = df.interpolate() # interploate() means it will give decimal values, as here most of the values are empty hence giving error
new_df


# In[21]:


new-df = df.interpolate(method='time') # check the temprature column
new_df


# In[ ]:


new_df = df.dropna() # will drop all values if anywhere NAN is present


# In[ ]:


new_df = df.dropna(how='all') # if entire row is having NAN values then it will drop the row


# In[22]:


new_df = df.dropna(thresh=1)


# In[23]:


new_df


# In[24]:


# as here date started from 1 and then next 4 so in between there is a gap. hence we can specify the range
dt = pd.date_range("01-01-2017","01-11-2017")
idx = pd.DatetimeIndex(dt)
df.reindex(idx)


# In[ ]:




