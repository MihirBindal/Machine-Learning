# -*- coding: utf-8 -*-
"""

@author: Win 10
"""

# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

'''from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])'''


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

plt.scatter(x_train,y_train)
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Salary vs Experience(Training set)")
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test,y_test)
plt.plot(x_test,regressor.predict(x_test),color='red')
plt.title("Salary vs Experience(Training set)")
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()