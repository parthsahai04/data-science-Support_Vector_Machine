#!/usr/bin/env python
# coding: utf-8

# In[37]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


iris = sns.load_dataset('iris')


# In[39]:


iris.head()


# In[40]:


iris.info()


# In[41]:


iris.describe()


# In[42]:


sns.pairplot(iris,hue='species')


# In[43]:


setosa = iris[iris['species']== 'setosa']
sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma')


# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# In[46]:


from sklearn.svm import SVC


# In[47]:


svc_model = SVC()


# In[48]:


svc_model.fit(X_train,y_train)


# In[49]:


predictions = svc_model.predict(X_test)


# In[50]:


from sklearn.metrics import classification_report,confusion_matrix 


# In[51]:


print(confusion_matrix(y_test,predictions))


# In[52]:


print(classification_report(y_test,predictions))


# In[54]:


from sklearn.model_selection import  GridSearchCV


# In[55]:


param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}


# In[56]:


grid = GridSearchCV(SVC(),param_grid,verbose=2)
grid.fit(X_train,y_train)


# In[57]:


grid_predictions = grid.predict(X_test)


# In[58]:


print(confusion_matrix(y_test,grid_predictions))


# In[59]:


print(classification_report(y_test,grid_predictions))


# In[ ]:




