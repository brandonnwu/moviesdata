import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

df = pd.read_csv('logistics1.1.csv')



#drop all unnecessary columns
df.drop(df.columns[[0,90,91]],axis=1,inplace=True)

#checking null value
# print(df.isnull().sum())

#separate target variable666
pd.set_option('display.max_columns',None)
print(df.head())
X=df.iloc[:,1:89] #get all the column except for y
y=df.iloc[:,89]
print(X,y)
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.3,random_state=0) #splitting data into 70/30

# #fitting regression
logReg = LogisticRegression() #fitting the data
logReg.fit(X_train,y_train)
y_pred=logReg.predict(X_test)

# 9.	Determine the accuracy of your predictions for survivability. (2)
cnf_matrix= metrics.confusion_matrix(y_test,y_pred) #checking accuracy with confusion matrix
print(metrics.accuracy_score(y_test,y_pred)) #printing accuracy

# 10.	Determine the confusion matrix. (2)
print(cnf_matrix) #printing confusion matrix







