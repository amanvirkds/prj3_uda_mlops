import requests
import json


data = {"age": 39, 
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "educationnum": 13,
        "maritalstatus": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capitalgain": 2174,
        "capitalloss": 0,
        "hoursperweek": 40,
        "nativecountry": "United-States"}

r = requests.post(
    "http://127.0.0.1:8000/classify/",
    data=json.dumps(data))


print(r.json()["salary"] )