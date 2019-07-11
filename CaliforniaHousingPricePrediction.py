#------------------------------------------------------------------------#
'''          Project 4: California Housing Price Prediction         '''
#------------------------------------------------------------------------#

# Step1: Import all libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step2: Load the data

# Step2.1: Read the “housing.csv” file from the folder into the program

housingData = pd.read_csv('housing.csv')

# Step2.2: Print first few rows of this data

print('Print first few rows of this data - ')
print()
print(housingData.head())

# Step2.3: Extract input (X) and output (y) data from the datase

X = housingData.iloc[:, :-1].values
y = housingData.iloc[:, [-1]].values

# Step3: Handle missing values: 
# Fill the missing values with the mean of the respective column

from sklearn.preprocessing import Imputer
missingValueImputer = Imputer()
X[:, :-1] = missingValueImputer.fit_transform(X[:, :-1])
y = missingValueImputer.fit_transform(y)

# Step4: Encode categorical data: 
# Convert categorical column in the dataset to numerical data

from sklearn.preprocessing import LabelEncoder
X_labelencoder = LabelEncoder()
X[:, -1] = X_labelencoder.fit_transform(X[:, -1])

# Step5: Split the dataset: Split the data into 
# 80% training dataset and 20% test dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Step6: Standardize data: Standardize training and test datasets

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

####################################################################
'''Task1: Perform Linear Regression'''
####################################################################

# Task1.1: Perform Linear Regression on training data

from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)

# Task1.2: Predict output for test dataset using the fitted model

predictionLinear = linearRegression.predict(X_test)

# Task1.3: Print root mean squared error (RMSE) from Linear Regression

from sklearn.metrics import mean_squared_error
mseLinear = mean_squared_error(y_test, predictionLinear)
print('Root mean squared error (RMSE) from Linear Regression = ')
print(mseLinear)

####################################################################
'''Task2: Perform Decision Tree Regression'''
####################################################################

# Task2.1: Perform Decision Tree Regression on training data

from sklearn.tree import DecisionTreeRegressor
DTregressor = DecisionTreeRegressor()
DTregressor.fit(X_train, y_train)

# Task2.2: Predict output for test dataset using the fitted model

predictionDT = DTregressor.predict(X_test)

# Task2.3: Print root mean squared error from Decision Tree Regression

from sklearn.metrics import mean_squared_error
mseDT = mean_squared_error(y_test, predictionDT)
print('Root mean squared error from Decision Tree Regression = ')
print(mseDT)

####################################################################
'''Task3: Perform Random Forest Regression'''
####################################################################

# Task3.1: Perform Random Forest Regression on training data

from sklearn.ensemble import RandomForestRegressor
RFregressor = RandomForestRegressor()
RFregressor.fit(X_train, y_train)

# Task3.2: Predict output for test dataset using the fitted model

predictionRF = RFregressor.predict(X_test)

# Task3.3: Print root mean squared error from Random Forest Regression

from sklearn.metrics import mean_squared_error
mseRF = mean_squared_error(y_test, predictionRF)
print('Root mean squared error from Random Forest Regression = ')
print(mseRF)

####################################################################
'''Task4: Bonus exercise: 
    Perform Linear Regression with one independent variable'''
####################################################################

# Task4.1: Extract just the median_income column from the 
# independent variables (from X_train and X_test)

X_train_median_income = X_train[: , [7]]
X_test_median_income = X_test[: , [7]]

# Task4.2: Perform Linear Regression to predict housing values 
# based on median_income

from sklearn.linear_model import LinearRegression
linearRegression2 = LinearRegression()
linearRegression2.fit(X_train_median_income, y_train)

# Task4.3: Predict output for test dataset using the fitted model

predictionLinear2 = linearRegression2.predict(X_test_median_income)

# Task4.4: Plot the fitted model for training data as well as 
# for test data to check if the fitted model satisfies the test data

# Task4.4.1: let us visualize the Training set

plt.scatter(X_train_median_income, y_train, color = 'green')
plt.plot (X_train_median_income, 
          linearRegression2.predict(X_train_median_income), color = 'red')
plt.title ('compare Training result - median_income / median_house_value')
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.show()

# Task4.4.2: let us visualize the Testing set

plt.scatter(X_test_median_income, y_test, color = 'blue')
plt.plot (X_train_median_income, 
          linearRegression2.predict(X_train_median_income), color = 'red')
plt.title ('compare Testing result - median_income / median_house_value')
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.show()

####################################################################
'''                          End                          '''
####################################################################