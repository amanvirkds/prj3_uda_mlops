# test_foo.py

from fastapi.testclient import TestClient
import json

from main import app

client = TestClient(app)


def test_get_path():
    """test will validate the home path of API"""
    
    r = client.get("/")
    assert r.status_code == 200
    
    
def test_classify_salary_1():
    
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

    r = client.post("/classify/", data=json.dumps(data))
    assert r.json()["salary"] == "<=50K"

    
def test_classify_salary_2():
    
    data = {"age": 37, 
            "workclass": "Private",
            "fnlgt": 280464,
            "education": "Some-college",
            "educationnum": 10,
            "maritalstatus": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "Black",
            "sex": "Male",
            "capitalgain": 0,
            "capitalloss": 0,
            "hoursperweek": 80,
            "nativecountry": "United-States"}

    r = client.post("/classify/", data=json.dumps(data))
    assert r.json()["salary"] == "<=50K"

