"""
Module: test_model_data_slicing.py
Author: Amandeep Singh
Date: 15-Jan-2023


Module will perform the model validations on each category of
categorical variables to ensure if model is performing well on
each slice.

"""

import pandas as pd
import pytest
import joblib
from src.ml.model import inference, compute_model_metrics
from src.ml.data import process_data



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

precision_threshold = .4
recall_threshold = .4
fbeta_threshold = .4

@pytest.fixture(autouse=True)
def data():
    """ funciton to read the data from csv file"""
    
    df = pd.read_csv("data/census.csv")
    df.columns = [col.strip() for col in df.columns]
    
    return df

@pytest.fixture
def model():
    """ funciton to read the data from csv file"""
    return joblib.load('model/model.pkl')

@pytest.fixture
def encoder():
    """ funciton to read the data from csv file"""
    return joblib.load('model/encoder.pkl')

@pytest.fixture
def lb():
    """ funciton to read the data from csv file"""
    return joblib.load('model/lb.pkl')
    
# content of test_example.py
def pytest_generate_tests(metafunc):
    """ function to create datasets on categorical variable slices
        for testing performance on each data slice
    """


    test_data = []
    df = pd.read_csv("data/census.csv")
    df.columns = [col.strip() for col in df.columns]
    for cat_feat in cat_features:
        for cat in df[cat_feat].unique():
            test_data.append(
                (cat_feat, 
                 cat, 
                 df.loc[df[cat_feat] == cat, :])
            )
    metafunc.parametrize("feat, category, cat_data", test_data)
    

def test_data_slicing(
        feat, category, cat_data, 
        data, model, encoder, lb):
    """function will perform the model validation on each data slice"""
    
    X_val, y_val, encoder, lb = process_data(
        cat_data, 
        categorical_features=cat_features, 
        label="salary", 
        training=False,
        encoder=encoder,
        lb=lb
    )

    val_preds = inference(model, X_val)
    precision, recall, fbeta = compute_model_metrics(y_val, val_preds)

            
    assert precision >= precision_threshold, f"Precision Failed: (Variable: {feat}, Category: {category}"
    assert recall >= recall_threshold, f"Recall Failed: (Variable: {feat}, Category: {category}"
    assert fbeta >= fbeta_threshold, f"FBeta Failed: (Variable: {feat}, Category: {category}"

    
    


            