#%% import the required libraries
 
import numpy as np
import pandas as pd 
import pickle


# Loading the data 
Data = pd.read_csv("diabetes2.csv")
# Seperate the dependent and independent variable
Y = Data.Outcome
X = Data.drop(['Outcome'],axis=1)

# Spliting the train-test data

from sklearn.model_selection import train_test_split

X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size = 0.2, random_state=5)

# Building the gradient boosting algorithm for predicting the diabetes

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score

Model = GradientBoostingClassifier(learning_rate= 0.213, n_estimators= 100,
                                   criterion='friedman_mse', min_samples_leaf= 13, 
                                   random_state=48, max_features='auto')

Model.fit(X_Train[['Age','Glucose','BMI']],Y_Train)
Y_Pred = Model.predict(X_Test[['Age','Glucose','BMI']])
cm = confusion_matrix(Y_Test,Y_Pred)
F1_score = f1_score(Y_Test,Y_Pred)
accuracy = accuracy_score(Y_Test,Y_Pred)
print(cm)
print(F1_score)
#%%  1st way
# Saving the model 

pickle.dump(Model,open('Model.pkl','wb'))
#%%  Loading the model after some time and predicting the result

#Loaded_Model = pickle.load(open('Model.pkl','rb'))
#Prediction = Loaded_Model.predict(X_Test[['Age','Glucose','BMI']])

