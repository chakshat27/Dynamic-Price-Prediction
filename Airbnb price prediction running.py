#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install missingno


# # Import Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as st
from scipy import stats
from scipy.stats import norm, skew #for some statistics

from sklearn import ensemble, tree, linear_model
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


# # Import Dataset

# In[3]:


data = pd.read_csv("file:///C:\\Users\\lenovo\\Downloads\\AB_NYC_2019.csv")


# # Explore the Dataset 

# In[4]:


data.sample(2)


# In[5]:


data.iloc[:,3:].describe()


# In[6]:


data.shape
#(rows,columns)


# # Checking for null values

# In[7]:


data.isnull().sum()


# In[8]:


(data.isnull().sum() / len(data)) *100
# % of null values


# # Cleaning Data

# Remove unnecessary columns

# In[9]:


data.drop(['name','id','host_name','last_review'] , axis=1 , inplace=True)


# fill null values in reviews_per_month by 0

# In[10]:


data['reviews_per_month'].fillna(0, inplace=True)
data.isnull().sum()


# In[11]:


data.head()
#display first 5 rows of dataset


# In[ ]:





# # Data Visualization

# In[14]:


# Plot a heatmap
numeric_data = data.select_dtypes(include=['number'])  # Select only numeric columns
corrmat = numeric_data.corr()  # Calculate correlation matrix from numeric data

plt.subplots(figsize=(12, 9))  # Set the figure size
sns.heatmap(corrmat, vmax=0.9, square=True, annot=True, cmap='coolwarm')  # Plot heatmap
plt.show()  # Display the plot


# Neighbourhood Group

# In[15]:


data['neighbourhood_group'].value_counts()


# In[16]:


ax = sns.countplot(x="neighbourhood_group", data=data)


# Neighbourhood

# In[33]:


df=pd.DataFrame(data['neighbourhood'].value_counts()).reset_index().rename(columns={'index': 'neighbourhood'})
df


# In[41]:


df = data.groupby('neighbourhood_group')['neighbourhood'].count().reset_index()
df.columns = ['neighbourhood_group', 'count']


fig = go.Figure(go.Bar(
    x=data['neighbourhood'],y=df['count'],
    marker={'color': df['count'], 
    'colorscale': 'Viridis'},  
    text=df['count'],
    textposition = "outside",
))
fig.update_layout(xaxis_title="Neighbourhood",yaxis_title="count")
fig.show()


# Room Type

# In[ ]:


ax = sns.countplot(x="room_type", data=data)


# Neighbourhood vs availability of room

# In[ ]:


plt.figure(figsize=(10,10))
ax = sns.boxplot(data=data, x='neighbourhood_group',y='availability_365')


# Room type vs availability

# In[19]:


plt.figure(figsize=(10,10))
ax = sns.boxplot(data=data, x='room_type',y='availability_365')


# Map of New York with neighbourhood groups

# In[20]:


fig = px.scatter(data, x='longitude', y='latitude',
                 color='neighbourhood_group') # Added color to previous basic 
fig.update_layout(xaxis_title="longitude",yaxis_title="latitude")
fig.show()


# #  Feature Engineering

# In[21]:


feature_columns=['neighbourhood_group','room_type','price','minimum_nights','calculated_host_listings_count','availability_365']


# In[22]:


all_data=data[feature_columns]
all_data.head()


# # Encoding categorical variables

# In[23]:


all_data['room_type']=all_data['room_type'].factorize()[0]
all_data['neighbourhood_group']=all_data['neighbourhood_group'].factorize()[0]
all_data.head()


# In[ ]:





# # Train Test Split

# In[24]:


y = all_data['price']
x= all_data.drop(['price'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)


# In[ ]:





# # Modelling

# Linear Regression

# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


linreg = LinearRegression()
linreg.fit(x_train,y_train)
y_pred=(linreg.predict(x_test))

print('R-squared train score: {:.3f}'.format(linreg.score(x_train, y_train)))
print('R-squared test score: {:.3f}'.format(linreg.score(x_test, y_test)))


# Ridge Regression

# In[26]:


from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(x_train, y_train)

y_pred=ridge.predict(x_test)

print('R-squared train score: {:.3f}'.format(ridge.score(x_train, y_train)))
print('R-squared test score: {:.3f}'.format(ridge.score(x_test, y_test)))


# Lasso Regression

# In[27]:


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=10,max_iter = 10000)
lasso.fit(x_train, y_train)

print('R-squared score (training): {:.3f}'.format(lasso.score(x_train, y_train)))
print('R-squared score (test): {:.3f}'.format(lasso.score(x_test, y_test)))


# Decision Tree Regressor

# In[28]:


from sklearn.tree import DecisionTreeRegressor
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)

print('R-squared score (training): {:.3f}'.format(DTree.score(x_train, y_train)))
print('R-squared score (test): {:.3f}'.format(DTree.score(x_test, y_test)))


# Random Forest Regressor

# In[29]:


from sklearn.ensemble import RandomForestClassifier

regressor = RandomForestClassifier()
regressor.fit(x_train, y_train)

print('R-squared score (training): {:.3f}'.format(regressor.score(x_train, y_train)))
print('R-squared score (test): {:.3f}'.format(regressor.score(x_test, y_test)))


# In[ ]:




