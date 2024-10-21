#!/usr/bin/env python
# coding: utf-8

# In[3]:


import random
import pandas as pd
import numpy as np
import seaborn as sns


# In[4]:


x1 = []
x2 = []
y = []
for i in range(0,20):
    x1.append(random.randint(1, 100))
    x2.append(random.randint(1, 100))
for i in range(0,20):
    if x1[i] > x2[i]:
        y.append(1)
    else:
        y.append(-1)
        
data = pd.DataFrame({'x1':x1, 'x2': x2, 'y':y})
print(x1)
print(x2)
print(y)


# In[9]:


def perceptron(xone, xtwo, y, a, epochs):
    #create random row vector of weights
    w = np.random.rand(1,2)
    #stores prediction sets
    ypreds = []
    #repeats for number of epochs
    for ind1 in range(0, epochs):
        #resets list of predicted ys
        ypred = []
        #repeats for number of x-y pairs
        for ind2 in range(0, len(xone)):
            #prediction = dot product of weights and transposed x vector
            pred = np.dot(w, np.transpose(np.array([xone[ind2], xtwo[ind2]])))
            #sign function; if prediction is greater than 0, add 1 to list of predictions. if not add -1
            if pred > 0:
                ypred.append(1)
            else:
                ypred.append(-1)
            #if prediction is not equal to actual y, edit weights
            if ypred[ind2] != y[ind2]:
                #new weights = weights + lr * actual y * non-transposed x vector
                w = w + (a*y[ind2] * (np.array([xone[ind2], xtwo[ind2]])))
            else:
                pass
            #at end of for loop, add list of predictions for this epoch to larger list for storage
            if ind2 == (len(x1) -1):
                ypreds.append(ypred)
            else:
                pass
        #return all recorded predictions and actual values
    return ypreds[epochs - 1], w
            


# In[10]:


perceptron(x1, x2, y, 1, 5)

