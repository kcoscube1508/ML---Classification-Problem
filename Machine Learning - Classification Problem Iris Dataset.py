#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Extract Features
 ##Features and target should not have null values
 ##Features should be numeric in nature
 ##Features should be of the type array/dataframes
 ##Features should have some rows and columns.

#Split the dataset into training and testing datasets
 ##Features should be on the same scale

#Train the model on the training dataset
#Test the moddel on the training dataset


# In[2]:


import pandas as pd
df=pd.read_csv("Iris.csv")


# In[3]:


df.head()


# In[4]:


df.Species.value_counts()


# In[5]:


df.describe()


# In[6]:


y=df["Species"]             #Target Variable


# In[7]:


X=df.drop("Species",axis=1)


# In[8]:


X.head()


# In[9]:


X=X.drop("Id",axis=1)


# In[10]:


X.head()


# In[11]:


X.isna().sum()


# In[12]:


X.dtypes


# In[13]:


type(X)


# In[14]:


X.shape


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42,stratify=y)


# In[17]:


y_test.value_counts()


# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


scaler=MinMaxScaler()


# In[20]:


X_train=scaler.fit_transform(X_train)


# In[21]:


X_test=scaler.transform(X_test)


# In[29]:


X_test


# In[30]:


from sklearn.neighbors import KNeighborsClassifier


# In[31]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[32]:


knn.fit(X_train,y_train)


# In[34]:


knn.score(X_train,y_train)        ## To check Accuracy


# In[33]:


knn.score(X_test,y_test)          ##To check Accuracy


# In[35]:


### Accuracy is very good in Training and Testing both, so this model can be considered as a Best fit.


# In[ ]:




