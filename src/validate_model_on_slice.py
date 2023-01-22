"""
Module: validate_model_on_slice.py
Author: Amandeep Singh
Date: 17-Jan-2023


Validate the model performance on slices of categorical features
by calculating the model metrices for each slice

"""


import pandas as pd
from ml.model import inference, compute_model_metrics
from ml.data import process_data

def model_metrics_on_slicing(
        feature, data, model, encoder=None, lb=None, cat_features=None):
    """function will perform the model validation on each data slice"""
    
    model_results_list = []
    for category in data[feature].unique():
        cat_data = data.loc[data[feature] == category, :]

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
        model_results_list.append([
            feature, category, precision, recall, fbeta
        ])
        
    model_metrices_df = pd.DataFrame(
        model_results_list, 
        columns=["Feature", "Category", "Precision", "Recall", "FBeta"])
    
    model_metrices_df.to_csv("output/" + feature + "_slice_output.txt", index=False)
    
    

            
