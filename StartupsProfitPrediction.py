#multiple linear regression

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('50_Startups.csv')

#set independent and dependent matrix
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#categorical data encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X =X[:,1:]        #take all columns of x from index 1 to end (don't include index 0 column)
#however here python linear regression lib takes care of the trap and we need not do it manually 

#training testing splitting of dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""    #no need to scale y

#fittin mlr model to dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting test set results
y_pred = regressor.predict(X_test)

#preparing the matrix of features for backward elimination ie. adding the intercept which isnot added by default
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1) #axis=1 means columns while 0 is rows

#optimising the model using backward elimination 
import statsmodels.formula.api as sm
X_opt = X[:, [0, 1, 2, 3, 4, 5]] #X-opt is optimatl matrix of feaures
regressor_OLS= sm.OLS(endog= y , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]] #X-opt is optimatl matrix of feaures
regressor_OLS= sm.OLS(endog= y , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]] #X-opt is optimatl matrix of feaures
regressor_OLS= sm.OLS(endog= y , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]] #X-opt is optimatl matrix of feaures
regressor_OLS= sm.OLS(endog= y , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3,]] #X-opt is optimatl matrix of feaures
regressor_OLS= sm.OLS(endog= y , exog = X_opt).fit()
regressor_OLS.summary()



