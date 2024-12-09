# test_app.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_temperature():
    test_input = {
        "city": "Delhi",
        "latitude": 28.6139,
        "longitude": 77.2090
    }
    
    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    
    predictions = response.json()
    assert isinstance(predictions, list)
    assert len(predictions) == 8  # 24 hours with 3-hour intervals
    
    for prediction in predictions:
        assert "datetime" in prediction
        assert "temperature" in prediction
        assert isinstance(prediction["temperature"], float)

def test_invalid_coordinates():
    test_input = {
        "city": "Invalid",
        "latitude": 200,  # Invalid latitude
        "longitude": 200  # Invalid longitude
    }
    
    response = client.post("/predict", json=test_input)
    assert response.status_code == 400