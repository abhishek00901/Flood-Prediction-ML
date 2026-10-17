from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("flood_model.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Flood Prediction</title>
</head>
<body>
    <h1>Flood Prediction System</h1>

    <form method="POST">

        Latitude:
        <input name="Latitude"><br><br>

        Longitude:
        <input name="Longitude"><br><br>

        Rainfall (mm):
        <input name="Rainfall (mm)"><br><br>

        Temperature (°C):
        <input name="Temperature (°C)"><br><br>

        Humidity (%):
        <input name="Humidity (%)"><br><br>

        River Discharge (m³/s):
        <input name="River Discharge (m³/s)"><br><br>

        Water Level (m):
        <input name="Water Level (m)"><br><br>

        Elevation (m):
        <input name="Elevation (m)"><br><br>

        Land Cover:
        <select name="Land Cover">
            <option>Water Body</option>
            <option>Forest</option>
            <option>Agricultural</option>
            <option>Desert</option>
            <option>Urban</option>
        </select><br><br>

        Soil Type:
        <select name="Soil Type">
            <option>Clay</option>
            <option>Peat</option>
            <option>Loam</option>
            <option>Sandy</option>
            <option>Silt</option>
        </select><br><br>

        Population Density:
        <input name="Population Density"><br><br>

        Infrastructure:
        <select name="Infrastructure">
            <option value="0">0</option>
            <option value="1">1</option>
        </select><br><br>

        Historical Floods:
        <select name="Historical Floods">
            <option value="0">0</option>
            <option value="1">1</option>
        </select><br><br>

        <button type="submit">Predict Flood</button>
    </form>

    {% if result is not none %}
        <h2>Flood Prediction: {{ result }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        data = {
            "Latitude": float(request.form["Latitude"]),
            "Longitude": float(request.form["Longitude"]),
            "Rainfall (mm)": float(request.form["Rainfall (mm)"]),
            "Temperature (°C)": float(request.form["Temperature (°C)"]),
            "Humidity (%)": float(request.form["Humidity (%)"]),
            "River Discharge (m³/s)": float(request.form["River Discharge (m³/s)"]),
            "Water Level (m)": float(request.form["Water Level (m)"]),
            "Elevation (m)": float(request.form["Elevation (m)"]),
            "Land Cover": request.form["Land Cover"],
            "Soil Type": request.form["Soil Type"],
            "Population Density": float(request.form["Population Density"]),
            "Infrastructure": int(request.form["Infrastructure"]),
            "Historical Floods": int(request.form["Historical Floods"])
        }

        df = pd.DataFrame([data])

        prediction = model.predict(df)
        result = prediction[0]

        if result == 1:
            result = "Flood Likely"
        else:
            result = "No Flood"

    return render_template_string(HTML, result=result)


if __name__ == "__main__":
    app.run(debug=True)