#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 21:41:01 2018

@author: neel
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/neel/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 1 - Artificial Neural Networks (ANN)/Section 4 - Building an ANN/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Taking Care of Categorial Variable will encode the data before we split it
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X= X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling and Data is pre-processed successfully
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Making of ANN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

#initialize the neural network and created an object of the sequential class
Classifier = Sequential()

#built the layers of  neural network i.e. first hidden layer with dropout
Classifier.add(Dense(output_dim=6, init='uniform',activation ='relu',input_dim =11))
Classifier.add(Dropout(p=0.1))

# built second hidden layer and we don't need input nodes as its already specified
Classifier.add(Dense(output_dim=6, init='uniform',activation ='relu'))
Classifier.add(Dropout(p=0.1))

# Adding the output Layer 
Classifier.add(Dense(output_dim=1, init='uniform',activation ='sigmoid'))

# Compiling the ANN
Classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting ANN classifier to the Training set
Classifier.fit(X_train,y_train, batch_size=10,nb_epoch=100)
# Create your classifier here

# Predicting the Test set results
y_pred = Classifier.predict(X_test)
y_pred =(y_pred>0.5)
