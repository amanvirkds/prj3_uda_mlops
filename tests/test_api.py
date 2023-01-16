# test_foo.py

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_path():
    """test will validate the home path of API"""
    
    r = client.get("/")
    assert r.status_code == 200