# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:13:41 2020

@author: Nathan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r'file_path_here')
df


df['Close'].fillna((df['Close'].mean()), inplace=True)

df.drop (['Date','High','Low','Open','Volume','Adj Close'], 1, inplace = True)

df.head()

prediction_days = 30  #n = 30 days


#Create another column (the target or dependent variable) shifted 'n' units up
df['Prediction'] = df[['Close']].shift(-prediction_days)


df.head()


df.tail()


# Convert the dataframe to a numpy array and drop the prediction column
X = np.array(df.drop(['Prediction'],1))

#Remove the last 'n' rows where 'n' is the prediction_days
X= X[:len(df)-prediction_days]
print(X)


# Convert the dataframe to a numpy array (All of the values including the NaN's) 
y = np.array(df['Prediction'])  
# Get all of the y values except the last 'n' rows 
y = y[:-prediction_days] 
print(y)


# Split the data into 80% training and 20% testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Set prediction_days_array equal to the last 30 rows of the original data set from the price column
prediction_days_array = np.array(df.drop(['Prediction'],1))[-prediction_days:]
print(prediction_days_array)


from sklearn.svm import SVR
# Create and train the Support Vector Machine 
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)#Create the model
svr_rbf.fit(x_train, y_train) #Train the model


# Testing Model: Score returns the accuracy of the prediction. 
# The best possible score is 1.0
svr_rbf_confidence = svr_rbf.score(x_test, y_test)
print("svr_rbf accuracy: ", svr_rbf_confidence)


# Print the predicted value
svm_prediction = svr_rbf.predict(x_test)
print(svm_prediction)

print()

#Print the actual values
print(y_test)



# Print the model predictions for the next 'n=30' days
svm_prediction = svr_rbf.predict(prediction_days_array)
print(svm_prediction)


plt.plot(svm_prediction)
plt.show

