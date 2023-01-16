"""
Module: train_model.py
Author: Amandeep Singh
Date: 15-Jan-2023


Module to perform the model training

"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    
    param_grid = { 
        'n_estimators': [100, 200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4, 5, 100],
        'criterion' :['gini', 'entropy']
    }
    
    rf_model = RandomForestClassifier(
            random_state=42)
    cv_model = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        verbose=False)
    rf_model.fit(X_train, y_train)
    #cv_model.fit(X_train, y_train)

    #return cv_model.best_estimator_
    return rf_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

            
