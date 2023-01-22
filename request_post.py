import requests
import json

data = {"age": 31, 
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "educationnum": 14,
        "maritalstatus": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capitalgain": 14084,
        "capitalloss": 0,
        "hoursperweek": 50,
        "nativecountry": "United-States"}

r = requests.post(
    "https://salary-prediction.herokuapp.com/classify/",
    data=json.dumps(data)
)

pred_val = r.json()["salary"]
print(f"Status code returned by API request is {r.status_code}")
print(f"Predicted value from API is {pred_val}")