#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('titanic_train_1.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df['Survived'].unique()


# In[6]:


df.drop(['PassengerId', 'Name', 'Ticket', "Fare"], inplace=True, axis=1)


# In[8]:


df['Age'].unique()


# In[9]:


df['Gender'].value_counts()


# In[10]:


df['Survived'].value_counts()


# In[11]:


sns.countplot(data=df, x='Survived')


# In[12]:


sns.countplot('Survived', data=df, hue='Gender')


# In[13]:


sns.countplot(data=df, x='Survived', hue='Pclass')


# In[18]:


df.head()


# In[19]:


df.info()


# In[20]:


df.isnull()


# In[21]:


sns.heatmap(df.isnull())


# In[22]:


df.drop(['Cabin'], axis=1, inplace=True)


# In[23]:


sns.heatmap(df.isnull())


# In[24]:


def inputAge(cols):
    Age = cols[0]
    Pclass = cols[0]
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[25]:


df['AgeNew'] = df[['Age','Pclass']].apply(inputAge, axis=1)


# In[26]:


df.head(10)


# In[27]:


df.isnull()


# In[28]:


sns.heatmap(df.isnull())


# In[29]:


df.drop(['Age'], axis=1, inplace=True)


# In[30]:


df.head()


# In[31]:


sns.pairplot(data=df)


# In[32]:


sns.heatmap(df.corr())


# In[ ]:




