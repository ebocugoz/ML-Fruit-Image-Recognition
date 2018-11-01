#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:18:40 2017

@author: erdembocugoz
"""

import chardet
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("Latest-Fruit-Train.csv")
df_test= pd.read_csv("Fruits_Test-NoLabels-test.csv")

        

features_test = list(df_test.columns[:1024])
X_test2= df_test[features_test]

features = list(df.columns[:1024])
X = df[features]
y = df["ClassId"]


dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
clf = RandomForestClassifier()       
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

knc = KNeighborsClassifier(1)
min_max=MinMaxScaler()
X_minmax=min_max.fit_transform(X)
knc.fit(X_minmax,y)

cross_score = cross_val_score(knc,X,y,cv=10)
Xtest2_minmax=min_max.fit_transform(X_test2)
predictions = knc.predict(Xtest2_minmax)
df_predict = pd.DataFrame({'ClassId': predictions})
writer = pd.ExcelWriter('predictionSes.xlsx', engine='xlsxwriter')
df_predict.to_excel(writer, sheet_name='Sheet1')
writer.save()