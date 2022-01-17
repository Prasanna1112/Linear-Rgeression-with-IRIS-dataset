#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


from sklearn import datasets
iris = datasets.load_iris()
# print(iris.DESCR)


# In[9]:


targ = iris.target
# print(targ)
# print(iris.target_names)


# In[20]:


data = iris.data
# print(data)


# In[22]:


from sklearn.model_selection import train_test_split

data_train,data_test,targ_train,targ_test=train_test_split(data,targ,test_size=0.2,random_state=101)


# In[25]:


from sklearn.linear_model import LinearRegression

LRclf = LinearRegression()
LRclf.fit(data_train,targ_train)


# In[26]:


targ_pred = LRclf.predict(data_test)
targ_pred


# In[27]:


# import matplotlib.pyplot as plt

#sepal length
# plt.scatter(data_test[:,0],targ_pred)
# plt.show()


# In[28]:


#sepal width
# plt.scatter(data_test[:,1],targ_pred)
# plt.show()


# In[29]:


#petal length
# plt.scatter(data_test[:,2],targ_pred)
# plt.show()


# In[30]:


#petal width
# plt.scatter(data_test[:,3],targ_pred)
# plt.show()


# In[35]:


import matplotlib.pyplot as plt

fig = plt.figure(figsize = (15, 10))
plt.subplot(2, 2, 1)

plt.title('Sepal Length')
plt.scatter(data_test[:,0],targ_pred)

plt.subplot(2, 2, 2)
plt.title('Sepal Width')
plt.scatter(data_test[:,1],targ_pred)

plt.subplot(2, 2, 3)
plt.title('Petal Width')
plt.scatter(data_test[:,2],targ_pred)

plt.subplot(2, 2, 4)
plt.title('Petal Width')
plt.scatter(data_test[:,3],targ_pred)


# In[ ]:




