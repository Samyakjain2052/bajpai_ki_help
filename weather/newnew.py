import requests

response = requests.post("http://localhost:8000/predict", 
    json={
        "city": "Delhi",
        "latitude": 28.6139,
        "longitude": 77.2090
    }
)
predictions = response.json()
print(predictions)