#!/usr/bin/env python
# coding: utf-8

# ## KMeans Clustering Using Mall_Customers Dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


# In[2]:


os.chdir('E:\\Professionals\\Naresh IT data\\important\\Datasets')


# In[3]:


df=pd.read_csv('Mall_Customers.csv')


# In[4]:


df.head(2)


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


X=df.iloc[:,[3,4]]
X.head(3)


# In[8]:


from sklearn.cluster import KMeans


# ### Elbow Method

# In[9]:


sse=[]
k_rng=range(1,10)
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)
plt.plot(k_rng,sse) 


# #### Fitting KMeans to the Dataset

# In[10]:


km=KMeans(n_clusters=5)
km.fit(X)
y_pred=km.predict(X)
y_pred


# In[11]:


df['cluster']=y_pred


# In[12]:


df.head()


# In[13]:


df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
df4=df[df.cluster==3]
df5=df[df.cluster==4]


# In[14]:


centroid=km.cluster_centers_
centroid


# ### Visualization

# In[15]:


sns.scatterplot(df1['Annual Income (k$)'],df1['Spending Score (1-100)'],color='Red')
sns.scatterplot(df2['Annual Income (k$)'],df2['Spending Score (1-100)'],color='Green')
sns.scatterplot(df3['Annual Income (k$)'],df3['Spending Score (1-100)'],color='Blue')
sns.scatterplot(df4['Annual Income (k$)'],df4['Spending Score (1-100)'],color='Orange')
sns.scatterplot(df5['Annual Income (k$)'],df5['Spending Score (1-100)'],color='Purple')
sns.scatterplot(centroid[:,0],centroid[:,1],color='Black')
plt.show()


# ## KMeans Clustering Using University Dataset

# In[16]:


df=pd.read_csv('Universities.csv')
df.head(2)


# In[17]:


X=df.iloc[:,[1,5]]
X.head(2)


# #### Elbow

# In[18]:


sse=[]
k_rng=range(1,10)
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)
plt.plot(k_rng,sse)    
plt.show()


# In[19]:


km=KMeans(n_clusters=4)
km.fit(X)
y_pred=km.predict(X)
y_pred


# In[20]:


df['cluster']=y_pred
df.head(3)


# In[21]:


df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
df4=df[df.cluster==3]


# In[22]:


centroid=km.cluster_centers_
centroid


# In[23]:


sns.scatterplot(df1['SAT'],df1['Expenses'],color='Red')
sns.scatterplot(df2['SAT'],df2['Expenses'],color='Green')
sns.scatterplot(df3['SAT'],df3['Expenses'],color='Blue')
sns.scatterplot(df4['SAT'],df4['Expenses'],color='Purple')
sns.scatterplot(centroid[:,0],centroid[:,1],color='Black')


# ## KMeans Clustering Using Income Dataset

# In[24]:


os.chdir('E:\\prasad\\practice\\dataset')


# In[25]:


df=pd.read_csv('income.csv')


# In[26]:


df.head()


# In[27]:


df.isnull().sum()


# In[28]:


df.shape


# In[29]:


df.head()


# In[30]:


sns.scatterplot('Age','Income($)',data=df)


# #### Feature Scalling

# In[31]:


from sklearn.preprocessing import StandardScaler


# In[32]:


sc=StandardScaler()


# In[33]:


df['Age']=sc.fit_transform(df[['Age']])
df['Income($)']=sc.fit_transform(df[['Income($)']])


# In[34]:


df.head()


# In[35]:


X=df.iloc[:,[1,2]]
X.head()


# #### Create Elbow Graph

# In[36]:


sse=[]
k_rng=range(1,10)
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)
plt.plot(k_rng,sse)    


# #### Use KMeans Algorithm

# In[37]:


km=KMeans(n_clusters=3)
km.fit(X)
y_pred=km.predict(X)
y_pred


# In[38]:


df['cluster']=y_pred
df.head()


# In[39]:


centroid=km.cluster_centers_
centroid


# In[40]:


df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]


# In[41]:


sns.scatterplot(df1['Age'],df1['Income($)'],color='Red')
sns.scatterplot(df2['Age'],df2['Income($)'],color='Green')
sns.scatterplot(df3['Age'],df3['Income($)'],color='Blue')
sns.scatterplot(centroid[:,0],centroid[:,1],color='Black')


# ## KMeans Clustering Using Iris Dataset

# In[42]:


from sklearn.datasets import load_iris


# In[43]:


iris=load_iris()


# In[44]:


dir(iris)


# In[45]:


df=pd.DataFrame(iris.data,columns=iris.feature_names)


# In[46]:


df['target']=iris.target


# In[47]:


df.head(2)


# In[48]:


sns.pairplot(df[['sepal length (cm)','sepal width (cm)']])


# In[49]:


sns.pairplot(df[['petal length (cm)','petal width (cm)']])


# In[50]:


df.head(2)


# In[51]:


X=df.iloc[:,[0,1]]
X.head(2)


# In[52]:


y=df.iloc[:,[2,3]]
y.head(2)


# #### Apply KMeans Clustering on sepal length (cm) & sepal width (cm)

# In[53]:


sse=[]
k_rng=range(1,10)
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)
plt.plot(k_rng,sse)    
plt.show()


# In[54]:


# Use 5 Clusters
km=KMeans(n_clusters=5)
km.fit(X)
y_pred=km.predict(X)
y_pred


# In[55]:


df['sepal_cluster']=y_pred
df.head()


# In[56]:


df1=df[df.sepal_cluster==0]
df2=df[df.sepal_cluster==1]
df3=df[df.sepal_cluster==2]
df4=df[df.sepal_cluster==3]
df5=df[df.sepal_cluster==4]


# In[57]:


centroid=km.cluster_centers_
centroid


# #### Visualization

# In[58]:


sns.scatterplot(df1['sepal length (cm)'],df1['sepal width (cm)'],color='Red')
sns.scatterplot(df2['sepal length (cm)'],df2['sepal width (cm)'],color='Green')
sns.scatterplot(df3['sepal length (cm)'],df3['sepal width (cm)'],color='Blue')
sns.scatterplot(df4['sepal length (cm)'],df4['sepal width (cm)'],color='Orange')
sns.scatterplot(df5['sepal length (cm)'],df5['sepal width (cm)'],color='Purple')
sns.scatterplot(centroid[:,0],centroid[:,1],color='Black')
plt.show()


# #### Apply KMeans Clustering on petal length (cm) & petal width (cm)

# In[59]:


sse=[]
k_rng=range(1,10)
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(y)
    sse.append(km.inertia_)
plt.plot(k_rng,sse)    
plt.show()


# In[60]:


# Use 3 Clusters
km=KMeans(n_clusters=3)
km.fit(y)
y_pred=km.predict(y)
y_pred


# In[61]:


df['petal_cluster']=y_pred
df.head()


# In[62]:


df1=df[df.petal_cluster==0]
df2=df[df.petal_cluster==1]
df3=df[df.petal_cluster==2]


# In[63]:


centroid=km.cluster_centers_
centroid


# In[64]:


sns.scatterplot(df1['petal length (cm)'],df1['petal width (cm)'],color='Green')
sns.scatterplot(df2['petal length (cm)'],df2['petal width (cm)'],color='Red')
sns.scatterplot(df3['petal length (cm)'],df3['petal width (cm)'],color='Orange')
sns.scatterplot(centroid[:,0],centroid[:,1],color='Black')
plt.show()


# In[ ]:




