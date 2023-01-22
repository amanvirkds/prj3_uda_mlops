# test_foo.py

from fastapi.testclient import TestClient
import json

from main import app

client = TestClient(app)


def test_get_path():
    """test will validate the home path of API"""
    
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["message"] == "Welcome, to the Salary Classification model"
    
    
def test_classify_low_salary():
    """test case 1 for salary prediction"""
    
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
    assert r.status_code == 200

    
def test_classify_high_salary():
    """test case 2 for salary prediction"""
    
    data = {"age": 58, 
            "workclass": "Self-emp-inc",
            "fnlgt": 210563,
            "education": "HS-grad",
            "educationnum": 9,
            "maritalstatus": "Married-civ-spouse",
            "occupation": "Sales",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capitalgain": 15024,
            "capitalloss": 0,
            "hoursperweek": 35,
            "nativecountry": "United-States"}

    r = client.post("/classify/", data=json.dumps(data))
    assert r.json()["salary"] == ">50K"
    assert r.status_code == 200
