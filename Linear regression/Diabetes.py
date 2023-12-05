#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np #here numpy is a powerful python library which is used for dealing with matrices and some mathematical function
import pandas as pd #here pandas used for data preprocessing and creating a datafrome
import plotly.express as px #here plotly is a high level interface for providing for ploting the given data in a graphs
import plotly.graph_objects as go
import plotly.io as pio
pio.templates
import seaborn as sns #here seaborn is used for data visualization and it is used for statiscal plots
import matplotlib.pyplot as plt #matplotlib.pyplot is used for scatter and some other graphs
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


from sklearn.datasets import load_diabetes #here we use sklearn datasets and here our project is about finding diabetes
load_diabetes=load_diabetes() #here first step we did loading the dataset
X=load_diabetes.data #here data is input features
Y=load_diabetes.target # here target is the output varible or target variable
data=pd.DataFrame(X,columns=load_diabetes.feature_names) #here we did creating a dataframe
data["Diabetes"]=Y #give name to target variable
data.head() #it show the data


# In[11]:


print(load_diabetes.DESCR) #here we get decription about the data in the dataset


# In[13]:


print(data.shape) #here output is in the format of no of rows and no of columns


# In[15]:


print(data.dtypes) #here we get what type of variables in the dataset


# In[17]:


data.info() #here it gives information about the data


# In[19]:


data.isnull().sum() #here we find the no of non empty values


# In[21]:


data.describe() #here we use see count and mean and standard deviation and min and max about the variables


# In[23]:


sns.pairplot(data,height=2.5) #here pairplot is kind of scatter plot it is used for data visuliaztion in eda(exploratory data analysis) and it is maniy used for to make relationship between variables
plt.tight_layout()


# In[29]:


sns.distplot(data['Diabetes']); # it a distribution ploting library


# In[34]:


print("Skewness : %f"%data['Diabetes'].skew()) #measures assymerty
print("Kurtness : %f"%data['Diabetes'].kurt()) #the kurt is used for finding the outliers


# In[35]:


fig,ax=plt.subplots() #here we ploting age vs Diabetes
ax.scatter(x=data['age'],y=data['Diabetes'])
plt.ylabel("Diabetes",fontsize=13)
plt.xlabel("age",fontsize=13)
plt.show()


# In[36]:


fig,ax=plt.subplots() #here we ploting sex vs Diabetes
ax.scatter(x=data['sex'],y=data['Diabetes'])
plt.ylabel("Diabetes",fontsize=13)
plt.xlabel("sex",fontsize=13)
plt.show()


# In[37]:


fig,ax=plt.subplots() #here we ploting bmi vs Diabetes
ax.scatter(x=data['bmi'],y=data['Diabetes'])
plt.ylabel("Diabetes",fontsize=13)
plt.xlabel("bmi",fontsize=13)
plt.show()


# In[39]:


fig,ax=plt.subplots() #here we ploting bp vs Diabetes
ax.scatter(x=data['bp'],y=data['Diabetes'])
plt.ylabel("Diabetes",fontsize=13)
plt.xlabel("bp",fontsize=13)
plt.show()


# In[45]:


from scipy import stats
from scipy.stats import norm,skew #for some stastics
sns.distplot(data['Diabetes'],fit=norm);
(mean,std)=norm.fit(data['Diabetes'])
print('\n mean={:.2f} and std ={:.2f}\n'.format(mean,std))
plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f})'.format(mean, std)], loc='best')
plt.ylabel('Frequency')
plt.title('Diabetes Prediction')
fig=plt.figure()
res=stats.probplot(data['Diabetes'],plot=plt)
plt.show()


# In[ ]:





# In[46]:


data['Diabetes']=np.log1p(data['Diabetes'])
sns.distplot(data['Diabetes'],fit=norm);
(mean,std)=norm.fit(data['Diabetes'])
print('\n mean={:.2f} and std ={:.2f}\n'.format(mean,std))
plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f})'.format(mean, std)], loc='best')
plt.ylabel('Frequency')
plt.title('Diabetes Prediction')
fig=plt.figure()
res=stats.probplot(data['Diabetes'],plot=plt)
plt.show()


# In[48]:


plt.figure(figsize=(10,10))
cor=data.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.PuBu)
plt.show()


# In[50]:


cor_target=abs(cor['Diabetes']) #value of the correlation
relevant_features=cor_target[cor_target>0.2]
names=[index for index,value in relevant_features.items()]
names.remove('Diabetes')
print(names)
print(len(names))


# In[55]:


from sklearn.model_selection import train_test_split
X=data.drop('Diabetes',axis=1)
y=data['Diabetes']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[56]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_train.shape)


# In[57]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[59]:


predictions=lr.predict(x_test)
print('Actual value of the diabetes:-',y_test[0])
print('predicted value of the diabetes:-',predictions[0])


# In[61]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,predictions)
rmse=np.sqrt(mse)
print(rmse)

