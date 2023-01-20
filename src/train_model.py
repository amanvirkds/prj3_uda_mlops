# Script to train machine learning model.

"""
Module: train_model.py
Author: Amandeep Singh
Date: 15-Jan-2023


Module to execute required steps for model training

"""

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from validate_model_on_slice import model_metrics_on_slicing
import joblib

# Add code to load in the data.
data = pd.read_csv("data/census.csv")
data.columns = [col.strip() for col in data.columns]

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
for cat_feat in cat_features:
    data[cat_feat] = data[cat_feat].apply(lambda x: x.strip())

X_train, y_train, encoder, lb = process_data(
    train, 
    categorical_features=cat_features, 
    label="salary", 
    training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, 
    categorical_features=cat_features, 
    label="salary", 
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
model = train_model(X_train
                    , y_train
                    , model_type="cv"
)
train_preds = inference(model, X_train)
precision, recall, fbeta = compute_model_metrics(y_train, train_preds)

print("Model metrices on train data:")
print(f"Percision: {precision}")
print(f"Recall: {recall}")
print(f"fbeta: {fbeta}")


# Predicitons on test set
test_preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, test_preds)
print("Model metrices on test data:")
print(f"Percision: {precision}")
print(f"Recall: {recall}")
print(f"fbeta: {fbeta}")


# Validate model performance on data slices

# Performance on Education Variable
model_metrics_on_slicing(
    "education", 
    data, model, encoder, lb, cat_features
)

# Performance on Work Class Variable
model_metrics_on_slicing(
    "workclass", 
    data, model, encoder, lb, cat_features
)


joblib.dump(model, 'model/model.pkl')
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(lb, 'model/lb.pkl')
