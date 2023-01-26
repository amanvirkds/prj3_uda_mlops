from fastapi import FastAPI
import pandas as pd
import json
import joblib
from src.ml.model import inference
from src.ml.data import process_data

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

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

@app.get("/")
async def root():
    return {"message": "Welcome, to the Salary Classification model V1"}


from pydantic import BaseModel

# Declare the data object with its components and their type.
class TaggedItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    educationnum: int
    maritalstatus: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capitalgain: int
    capitalloss: int
    hoursperweek: int
    nativecountry: str
        

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/classify/")
async def create_item(item: TaggedItem):
    
    data = {"age": item.age, 
            "workclass": item.workclass,
            "fnlgt": item.fnlgt,
            "education": item.education,
            "education-num": item.educationnum,
            "marital-status": item.maritalstatus,
            "occupation": item.occupation,
            "relationship": item.relationship,
            "race": item.race,
            "sex": item.sex,
            "capital-gain": item.capitalgain,
            "capital-loss": item.capitalloss,
            "hours-per-week": item.hoursperweek,
            "native-country": item.nativecountry}
    
    df = pd.DataFrame.from_dict(
        data, orient='index').T
    
    model = joblib.load('model/model.pkl')
    encoder = joblib.load('model/encoder.pkl')
    lb = joblib.load('model/lb.pkl')
    
    X_val, y_val, encoder, lb = process_data(
        df, 
        categorical_features=cat_features,  
        training=False,
        encoder=encoder,
        lb=lb
    )

    val_preds = inference(model, X_val)
    
    salary = "<=50K"
    if int(val_preds) == 1:
        salary = ">50K"
    
    return {"salary": salary}