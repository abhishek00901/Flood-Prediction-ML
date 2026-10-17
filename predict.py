import joblib
import pandas as pd

model = joblib.load("flood_model.pkl")

data = {
    "Latitude": [25.36],
    "Longitude": [85.61],
    "Rainfall (mm)": [200],
    "Temperature (°C)": [30],
    "Humidity (%)": [80],
    "River Discharge (m³/s)": [500],
    "Water Level (m)": [5],
    "Elevation (m)": [100],
    "Land Cover": ["Urban"],
    "Soil Type": ["Loam"],
    "Population Density": [1000],
    "Infrastructure": [1],
    "Historical Floods": [1]
}

df = pd.DataFrame(data)

prediction = model.predict(df)

print("Flood Prediction:", prediction[0])