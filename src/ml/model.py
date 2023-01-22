"""
Module: train_model.py
Author: Amandeep Singh
Date: 15-Jan-2023


Module to perform the model training

"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd


# Optional: implement hyperparameter tuning.
def model_fit(X_train, y_train, model_type=None):
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
    
    
    if model_type == "rf":
        model = RandomForestClassifier(
                random_state=42)
        model.fit(X_train, y_train)
        return model
    elif model_type == "cv":
        param_grid = { 
            'n_estimators': [50, 100, 200, 500],
            'max_features': ['sqrt', 'log2', 2, 5, 10, 20],
            'max_depth' : [2, 4, 5, 6, 7, 9, 11, 20],
            'criterion' :['gini', 'entropy']
        }
        rf_model = RandomForestClassifier(
                random_state=42)
        model = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=10,
            verbose=False)
        model.fit(X_train, y_train)
        return model.best_estimator_
    else:
        model = LogisticRegression(
            solver='lbfgs',
            max_iter=3000,
            verbose=False)
        model.fit(X_train, y_train)
        return model
        
    


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

            
