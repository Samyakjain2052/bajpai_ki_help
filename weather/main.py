# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from weather_predictor import WeatherPredictor
from typing import List

app = FastAPI()
predictor = WeatherPredictor()

class LocationInput(BaseModel):
    city: str
    latitude: float
    longitude: float

class PredictionOutput(BaseModel):
    datetime: str
    temperature: float

@app.post("/predict", response_model=List[PredictionOutput])
async def predict_temperature(location: LocationInput):
    try:
        predictions = predictor.predict(
            location.city,
            location.latitude,
            location.longitude
        )
        return predictions
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}