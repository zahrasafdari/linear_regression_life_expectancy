#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv("Life_Expectancy_Data.csv")
df.head()


# In[7]:


df.columns = df.columns.to_series().apply(lambda x: x.strip())


# In[8]:


cdf = df[['Country','Year','Status','Adult_Mortality','infant_deaths','Life_expectancy','percentage_expenditure','Total_expenditure']]
df[pd.to_numeric(df['Life_expectancy'], errors='coerce').notnull()]
df[pd.to_numeric(df['Adult_Mortality'], errors='coerce').notnull()]
cdf.head(9)


# In[9]:


viz = cdf[['Adult_Mortality','infant_deaths','Life_expectancy','percentage_expenditure','Total_expenditure']]
pd.to_numeric(df.Adult_Mortality, errors='coerce').dropna()

pd.to_numeric(df.Life_expectancy, errors='coerce').dropna()
viz.hist()
plt.show()


# In[10]:


plt.scatter(cdf.Adult_Mortality, cdf.Life_expectancy,  color='blue')
plt.xlabel("Adult_Mortality")
plt.ylabel("Life_expectancy")
plt.show()


# In[11]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[12]:


plt.scatter(train.Adult_Mortality, train.Life_expectancy,  color='blue')
plt.xlabel("Adult_Mortality")
plt.ylabel("Life_expectancy")
plt.show()


# In[13]:


from sklearn import linear_model
regr = linear_model.LinearRegression()


train_x = np.asanyarray(train[['Adult_Mortality']])
train_y = np.asanyarray(train[['Life_expectancy']])
np.isnan(train_x).any() 
np.isnan(train_y).any()

train_x[np.isnan(train_x)] = np.median(train_x[~np.isnan(train_x)])
train_y[np.isnan(train_y)] = np.median(train_y[~np.isnan(train_y)])


regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[14]:


plt.scatter(train.Adult_Mortality, train.Life_expectancy,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Adult_Mortality")
plt.ylabel("Life_expectancy")


# In[15]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['Adult_Mortality']])
test_y = np.asanyarray(test[['Life_expectancy']])
test_x[np.isnan(test_x)] = np.median(test_x[~np.isnan(test_x)])
test_y[np.isnan(test_y)] = np.median(test_y[~np.isnan(test_y)])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )


# In[ ]:





# In[ ]:




