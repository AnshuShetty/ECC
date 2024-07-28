# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 21:10:01 2024

@author: Dell
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Load data
df = pd.read_csv("pima_indian.csv")

# Define feature columns and predicted class names
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

# Prepare features and target variable
X = df[feature_col_names].values
y = df[predicted_class_names].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print('Total number of Training Data:', y_train.shape)
print('Total number of Test Data:', y_test.shape)

# Train Naive Bayes (NB) classifier
clf = GaussianNB()
clf.fit(X_train, y_train)

# Predictions
predicted = clf.predict(X_test)
predict_test_data = clf.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Model Evaluation
print('\nConfusion matrix:')
print(metrics.confusion_matrix(y_test, predicted))

print('\nAccuracy of the classifier:', metrics.accuracy_score(y_test, predicted))

print('\nPrecision:', metrics.precision_score(y_test, predicted))

print('\nRecall:', metrics.recall_score(y_test, predicted))

print("\nPredicted Value for individual Test Data:", predict_test_data)