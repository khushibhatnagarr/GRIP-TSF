#!/usr/bin/env python
# coding: utf-8

# # Prediction using Supervised ML
# #
# # Author: Khushi Bhatnagar
# 
# #Predict the percentage of an student based on the no. of study hours.This is a simple linear regression task as it involves just 2 variables.

# In[31]:


#importing all the libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Reading the Data from link

# In[32]:


dataset = pd.read_csv("http://bit.ly/w-data")


# In[33]:


#the first five values in the dataset
dataset.head()


# In[34]:


dataset.info()

#The data has 25 observations with 2 columns therefore it suggests that this data is Simple linear regression
# In[35]:


#Hours Vs Percentage of Scores
plt.scatter(dataset['Hours'], dataset['Scores'])
plt.title('Hours vs Percentage')
plt.xlabel('Studied Hours')
plt.ylabel('Scores')
plt.show()

 #From the graph we can see that there is positive linear realation 
# Splitting data into feature and target

# In[36]:


#X will take all the values except for the last column which is our dependent variable (target variable)
X = dataset.iloc[:, :-1].values
X


# In[37]:


Y = dataset.iloc[:, -1].values
Y


# Training the Data

# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# Building model

# In[39]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# In[40]:


# Plotting the regression line and test data
line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, Y)
plt.plot(X, line,color = 'pink')
plt.show()


# #Predicting the Test set results

# In[41]:


Y_pred = regressor.predict(X_test)
print(Y_pred)


# #Visualising the Training set results
# 

# In[42]:


plt.scatter(X_train, Y_train, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Hours vs. Percentage (Training set)')
plt.xlabel('Hours studied')
plt.ylabel('Percentage of marks')
plt.show()


# In[43]:


#Visualising the Test set results
plt.scatter(X_test, Y_test, color = 'yellow')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hours vs. Percentage (Test set)')
plt.xlabel('Hours studied')
plt.ylabel('Percentage of marks')
plt.show()


# Comparing the actual values with the predicted ones.

# In[44]:


dataset = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
dataset


# # Pridicting score

# In[45]:


dataset = np.array(9.25)
dataset = dataset.reshape(-1, 1)
pred = regressor.predict(dataset)
print("If the student studies for 9.25 hours/day, the score is {}.".format(pred))


# Errors

# In[46]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))


# In[47]:


from sklearn.metrics import r2_score
print("The R-Square of the model is: ",r2_score(Y_test,Y_pred))


# # CONCLUSION
# If the student studies for 9.25 hours/day, the score is [93.69173249].

# 

# 

# 
