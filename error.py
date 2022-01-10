#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


ts1=np.loadtxt("ts1.dat")
ts2=np.loadtxt("ts2.dat")
ts3=np.loadtxt("ts3.dat")


# In[17]:


ts1.sum()/len(ts1)


# In[18]:


plt.plot(ts1)
plt.ylabel('some numbers')
plt.show()


# In[20]:


for i in range(2000):
  print(ts1[i])


# In[53]:


avg1 = np.average(ts1)
std1 = np.std(ts1)
astd1 = np.std(ats1)
avg2 = np.average(ts2)
std2 = np.std(ts2)
avg3 = np.average(ts3)
std3 = np.std(ts3)





# In[29]:


ats1 = ts1 - avg1

ats2 = ts2 - avg2
ats3 = ts3 - avg3


# In[32]:


plt.plot(ats3)
plt.ylabel('some numbers')
plt.show()


# In[59]:


autof1 = np.zeros(ts1.shape)
for t in range(2000):
 for i in range(2000):
  autof1[t] = autof1[t] + ats1[i]*ats1[(i+ t)%len(ts1)]

autof1 =autof1/autof1[0]


# In[60]:


plt.plot(autof1)
plt.xlim([0, 100])
plt.ylabel('some numbers')
plt.show()


# In[96]:


tt = np.zeros((20))
for i in range (20):
 tt[i] = i


# In[97]:


tt


# In[65]:


from pylab import *
from scipy.optimize import curve_fit


# In[76]:


tt.shape
autof1.shape


# In[98]:


def func(x,a):
    return np.exp(a*x)

popt, pcov = curve_fit(func,tt[0:20],autof1[0:20],1e-6 )


# In[104]:


1/popt[:]


# In[99]:


yy = func(tt, popt)


# In[101]:




plt.plot(autof1)
plt.plot(yy)
plt.xlim([0, 20])
plt.ylabel('some numbers')
plt.show()


# In[105]:


plt.plot(autof1)

plt.xlim([0, 20])
plt.ylabel('some numbers')
plt.show()


# In[128]:


ct =-1
er =np.zeros(9)


for bsize in range(10,100,10):
 ct =ct+1        
 numblocks =int(len(ts1)/bsize)
 bav=np.zeros((numblocks))
 for i in range(numblocks):
    bav[i]= np.average(ts1[0 +i*bsize : bsize+ i*bsize])
 er[ct] = np.std(bav)/np.sqrt(numblocks)


# In[129]:


st =np.zeros(9)
for i in range(9):
  st[i]=10*(i+1)


# In[130]:


er2 =np.zeros(9)
for i in range(9):
 er2[i] = sum(er[0:i+1])


# In[132]:



plt.plot(st,er)

plt.xlim([10, 100])
plt.ylabel('some numbers')
plt.show()


# In[ ]:




