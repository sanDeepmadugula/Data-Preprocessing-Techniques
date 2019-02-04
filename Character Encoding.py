#!/usr/bin/env python
# coding: utf-8

# In[1]:


# start's with a string
before = "This is the euro symbol: €"
type(before)


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


# encode it to a different encoding, replacing characters that raise errors
after = before.encode('utf-8', errors='replace')
type(after)


# In[4]:


after


# In[5]:


# converting it back to utf-8
print(after.decode('utf-8'))


# In[6]:


print(after.decode('ascii'))


# In[7]:


before = "This is the euro symbol: €"
after = before.encode('ascii',errors= 'replace')
print(after.decode('ascii'))


# # Reading file with encoding problems

# In[8]:


import os
os.chdir('D:/python using jupyter/Data Preprocessing')
kickstarter_2016 = pd.read_csv('ks-projects-201612.csv')


# In[12]:


import chardet
with open('ks-projects-201612.csv','rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
    
print(result)


# In[13]:


# as it is given in the form of 'Windows-1252' with confidence given 0.73
kickstarter_2016 = pd.read_csv('ks-projects-201612.csv',encoding='Windows-1252')


# In[14]:


kickstarter_2016.head(5)


# In[ ]:


# hence now it is reading properly

