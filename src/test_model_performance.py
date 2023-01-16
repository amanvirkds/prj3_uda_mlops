"""
Module: test_model_performance.py
Author: Amandeep Singh
Date: 15-Jan-2023


Module will perform the model validations by calculating the
model metrices.

"""

import pandas as pd
import pytest
import joblib
from ml.model import inference, compute_model_metrics
from ml.data import process_data

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

@pytest.fixture(autouse=True)
def data():
    """ funciton to read the data from csv file"""
    
    df = pd.read_csv("../data/census.csv")
    df.columns = [col.strip() for col in df.columns]
    
    return df

@pytest.fixture
def model():
    """ funciton to read the data from csv file"""
    return joblib.load('../model/model.pkl')

@pytest.fixture
def encoder():
    """ funciton to read the data from csv file"""
    return joblib.load('../model/encoder.pkl')

@pytest.fixture
def lb():
    """ funciton to read the data from csv file"""
    return joblib.load('../model/lb.pkl')
    
def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."
    
def test_model_performance(data, model, encoder, lb):
    """ test the model metrices values"""
    
    # Proces the test data with the process_data function.
    X_val, y_val, encoder, lb = process_data(
        data, 
        categorical_features=cat_features, 
        label="salary", 
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    val_preds = inference(model, X_val)
    precision, recall, fbeta = compute_model_metrics(y_val, val_preds)
    
    assert precision >= .9, "Model precision metric is less then .9"
    assert recall >= .9, "Model recall metric is less then .9"
    assert fbeta >= .9, "Model fbeta metric is less then .9"
    
    


            