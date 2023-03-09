# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:03:02 2021

@author: Shaun Tranter 19124456
"""

# -- Tool Imports--
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import GridSearchCV as Grid
import warnings
warnings.filterwarnings("ignore")

#-- Model Imports--
from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import BernoulliNB as B
from sklearn.naive_bayes import GaussianNB as G

#--Import dat set--
Data = pd.read_csv("healthcare-dataset-stroke-data.csv")

#--- Data Analysis---
Data.info()                    #Shows all column's count, null status and data types

Data.describe().T
print(Data.describe().T)       #Prints the mean, mode and max of each column with count

Data = Data.drop('id', axis=1) #Drops ID as an unneccesary feature 
print(Data.head())             #Shows info without ID

for i in Data.columns:          #A loop that counts the instances of the varibles in each column and prints them
    print(Data[i].value_counts())
    print("---------------------------------------")  
    
#--- Data Cleaning --- 
print(Data.isnull().sum())                                                                         #Sums up all the null values in the data set and prints
Data["bmi"].fillna(Data["bmi"].median(), inplace=True)                                             #Fills in missing values based on the median of the BMI column
print(Data.duplicated().sum())                                                                     #prints to show duplicated values which should be 0
print("---------------------------------------")

Data["smoking_status"].replace("Unknown", Data["smoking_status"].mode().values[0], inplace=True)   #Replaces all "unkown" values in smoking_status to the mode value in the data
print(Data["smoking_status"].value_counts())                                                       #prints smoke_status counts to show the change
print("---------------------------------------")

#--- Visualisation ---
categorical_features = ['gender', 'hypertension', 'heart_disease','ever_married', 'work_type', 'Residence_type', 'smoking_status'] 
num_features = ['age', 'avg_glucose_level', 'bmi']  #Seperating Types of columns to make it easier to visualise which will also play apart in standardizations


for f in categorical_features:
    print(Data[f].unique())        #A loop that prints the values of the catagorical features in the dataset 

for i in categorical_features:     #A loop that prints count graphs for all the catagorical freatures 
    plt.figure(figsize=(10,5))
    sns.countplot(x = Data[i])
    plt.show()
    
    
plt.figure(figsize=(10,6))
sns.distplot(Data["age"])              # plots age graph
plt.title("Age", size=15)
plt.show()
    
sns.pairplot(Data)                # Pair plotting showing how each feature correlates with the other
plt.show()
   
#--- Label Encoding ---
encoder = LabelEncoder()                                         #Turns the catagorical featues into indexs so its easier to go through
Ldata = Data[categorical_features].apply(encoder.fit_transform)  
print(Ldata.head())                                              #Shows how the catagorical fetaues have been turned into index numbers e.g. Male = 0 Female = 1
print("---------------------------------------")

y = Data["stroke"]                                                #Sets y as the target which is the stroke column 
x = pd.concat([Ldata,Data[num_features]], axis =1)                #Sets x as the other columns by putting the catagorcal and num features togther so all should be numbers
print(x.head())                                                   #Prints the table showing all column vaibles as ints and floats
print("---------------------------------------")

#--- Splitting the Data ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42) #Splits the data so the test size is 30% sets random split as 42

#--- Standardisation ---
scaler = StandardScaler()                      #Sets the scaler

x_train = scaler.fit_transform(x_train)        #Fits the training varibes in the scaler 
x_test = scaler.transform(x_test)              #scales the testing set of vaibles 

#--- Training the models ---
models = pd.DataFrame(columns=["Model","Accuracy Score"]) #sets models as a dataframe making it easier to present 

log_reg = LR()                                            #LogisticRegression Model Training
log_reg.fit(x_train, y_train)
LRpredictions = log_reg.predict(x_test)
score = accuracy_score(LRpredictions, y_test) * 100
print(f"LogisticRegression: {score}")
print("---------------------------------------")

new_row={"Model": "LogisticRegression", "Accuracy Score": score} #Adds score to a row of model dataframe
models = models.append(new_row, ignore_index=True)


Tree =tree.DecisionTreeClassifier()                       #DecisionTreeClassifier Model Training
Tree.fit(x_train, y_train)
Treepredictions = Tree.predict(x_test)
with open ("Stroke.dot", 'w') as f:
    f = tree.export_graphviz( Tree, out_file=f  )          #Exports graphiviz file
score = accuracy_score(Treepredictions, y_test) * 100
print(f"Decision Tree: {score}")
print("---------------------------------------")

new_row={"Model": "DecisionTreeClassifier", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

GNB = G()                                                 #GaussianNB Model Training
GNB.fit(x_train, y_train)
GNBpredictions = GNB.predict(x_test)
score = accuracy_score(GNBpredictions, y_test) * 100
print(f"GaussianNB: {score}")
print("---------------------------------------")

new_row={"Model": "GaussianNB", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

BNB = B()                                                #BernoulliNB Model Training
BNB.fit(x_train, y_train)
BNBpredictions = BNB.predict(x_test)
score = accuracy_score(BNBpredictions, y_test) * 100
print(f"BernoulliNB: {score}")
print("---------------------------------------")

new_row={"Model": "BernoulliNB", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

svm = SVC()                                              #SVC Model Training
svm.fit(x_train, y_train)
svmpredictions = svm.predict(x_test)
score = accuracy_score(svmpredictions, y_test) * 100
print(f"SVC: {score}")
print("---------------------------------------")

new_row={"Model": "SVC", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

randomforest = RFC(n_estimators=20, random_state=42)    #Random Forest Model Training n_estimators at 20
randomforest.fit(x_train, y_train)
RFCpredictions = randomforest.predict(x_test)
score = accuracy_score(RFCpredictions, y_test) * 100
print(f"RandomForestClassifier: {score}")
print("---------------------------------------")

new_row={"Model": "RandomForestClassifier", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)  

feat_importances = pd.Series(randomforest.feature_importances_, index=x.columns) #plots the importance of deciding features in random forest classifier
feat_importances.nlargest(20).plot(kind='barh')                       


print(models.sort_values(by="Accuracy Score", ascending= False))           #Prints out all models in the dataframe in decending order

# --- Validation ---
LRCM = CM(y_test, LRpredictions)
print("------------Logistic Regression-----------------")
print(LRCM)
print("-----------------------------")
TreeCM = CM(y_test, Treepredictions)
print("------------Decision Tree-----------------")
print(TreeCM)
print("-----------------------------")                                      # confusion matrix for each model
GNBCM = CM(y_test, GNBpredictions)
print("------------GaussianNB-----------------")
print(GNBCM)
print("-----------------------------")
print("------------BernoulliNB-----------------")
BNBCM = CM(y_test, BNBpredictions)
print(BNBCM)
print("-----------------------------")
SVMCM = CM(y_test, svmpredictions)
print("------------SVC--------------------------")
print(SVMCM )
print("-----------------------------")
RFCCM = CM(y_test, RFCpredictions)
print("------------Random Forest-----------------")
print(RFCCM)
print("-----------------------------")

LRMAE = MAE(y_test, LRpredictions)
print("------------Logistic Regression-----------------")
print(LRMAE)
print("-----------------------------")                        # mean absolute error for each model
TreeMAE = MAE(y_test, Treepredictions)
print("------------Decision Tree-----------------")
print(TreeMAE)
print("-----------------------------") 
GNBMAE = MAE(y_test, GNBpredictions)
print("------------GaussianNB-----------------")
print(GNBMAE)
print("-----------------------------")
BNBMAE = MAE(y_test, BNBpredictions)
print("------------BernoulliNB-----------------")
print(BNBMAE)
print("-----------------------------")
SVMMAE = MAE(y_test, svmpredictions)
print("------------SVC--------------------------")
print(SVMMAE)
print("-----------------------------")
RFCMAE = MAE(y_test, RFCpredictions)
print("------------Random Forest-----------------")
print(RFCMAE)
print("-----------------------------")

# --- Grid search cross validation for the highest scoring models ---

x = scaler.fit_transform(x)   # Scales the variables of x which was all data in numerical form before splitting the data

model_params = {"svc" : {"model": SVC(gamma ="auto"),"params":{ "C":[1,5,10],"kernel":["rbf","linear"]}}, 
                "LogisticRegression" : {"model": LR(solver ="liblinear",multi_class ="auto"),"params":{ "C":[1,5,10]}},  #Lists the Top three models of the other training and sets parameters to go through
                "RandomForestClassifer" : {"model": RFC(),"params":{ "n_estimators":[1,5,10,20]}}
                }

scores = []   #prepares a list of scores

for model_name, mp in model_params.items():
    classifier = Grid(mp["model"], mp["params"], cv =5, return_train_score = False)
    classifier.fit(x, y)
    scores.append({"model": model_name, "best score":classifier.best_score_,"best params":classifier.best_params_ }) #A loop going through each model and its scores to append them to scores list

df = pd.DataFrame(scores, columns =["model","best score","best params"])
print("---------------------------GridSearch-------------------------------------")         #prints the table of scores 
print(df)




