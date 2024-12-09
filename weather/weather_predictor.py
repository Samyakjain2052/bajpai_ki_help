# weather_predictor.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from meteostat import Hourly, Stations

class WeatherPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def get_weather_data(self, city, latitude, longitude, start_date, end_date):
        stations = Stations()
        stations = stations.nearby(latitude, longitude)
        station = stations.fetch(1)
        
        if station.empty:
            raise ValueError("No weather station found nearby")
            
        data = Hourly(station.index[0], start_date, end_date)
        data = data.fetch()
        return data
        
    def prepare_features(self, data, n_lags=8):
        df = data.copy()
        for lag in range(1, n_lags + 1):
            df[f'Temperature_Lag_{lag}'] = df['Temperature'].shift(lag)
        df.dropna(inplace=True)
        return df
        
    def predict(self, city, latitude, longitude):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        # Get and process data
        data = self.get_weather_data(city, latitude, longitude, start_date, end_date)
        if data.empty:
            raise ValueError("No data available")
            
        data = data[['temp']].rename(columns={'temp': 'Temperature'})
        data.index.name = 'DateTime'
        data = data.resample('3H').mean()
        data['Temperature'] = data['Temperature'].interpolate(method='time')
        data = data.dropna(subset=['Temperature'])
        data.reset_index(inplace=True)
        
        # Prepare features and train model
        data_prepared = self.prepare_features(data)
        train_data = data_prepared.iloc[:-8]
        
        X_train = train_data.drop(columns=['Temperature', 'DateTime'])
        y_train = train_data['Temperature']
        
        self.model.fit(X_train, y_train)
        
        # Predict next 24 hours
        future_times = pd.date_range(start=end_date, periods=8, freq='3H')
        predictions = []
        last_known_data = data_prepared.iloc[-1:].copy()
        
        for time in future_times:
            input_features = last_known_data.drop(columns=['Temperature', 'DateTime'])
            pred_temp = self.model.predict(input_features)[0]
            predictions.append({
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                'temperature': float(pred_temp)
            })
            
            new_row = pd.DataFrame({
                'DateTime': [time],
                'Temperature': [pred_temp]
            })
            
            for lag in range(1, 9):
                if lag == 1:
                    new_row[f'Temperature_Lag_{lag}'] = last_known_data['Temperature'].values
                else:
                    new_row[f'Temperature_Lag_{lag}'] = last_known_data[f'Temperature_Lag_{lag-1}'].values
                    
            last_known_data = new_row
            
        return predictions