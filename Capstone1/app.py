from flask import Flask, render_template, request
import joblib
import numpy as np
from flask_cors import CORS  # <-- import CORS

app = Flask(__name__)
CORS(app)

# Load model and numeric scaler
model = joblib.load("manufacturing_prediction_model2.pkl")  # trained on numeric + categorical
scaler = joblib.load("feature_scaler.pkl")                  # scales numeric features only

# Categorical options
shift_options = ["Day", "Evening", "Night"]
machine_options = ["Type_A", "Type_B", "Type_C"]
material_options = ["Standard", "Premium", "Economy"]
day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Numeric features (order must match training)
numeric_features = [
    "Injection_Temperature",
    "Injection_Pressure",
    "Cycle_Time",
    "Cooling_Time",
    "Material_Viscosity",
    "Machine_Age",
    "Operator_Experience",
    "Temperature_Pressure_Ratio",
    "Total_Cycle_Time",
    "Efficiency_Score"
]

@app.route("/")
def home():
    return render_template(
        "index.html",
        shift_options=shift_options,
        machine_options=machine_options,
        material_options=material_options,
        day_options=day_options,
        numeric_features=numeric_features
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Read JSON sent from frontend

    if not data:
        return {"error": "No input data received"}, 400

    try:
        # Extract numeric inputs
        inputs = [float(data[feature]) for feature in numeric_features]

        # Extract categorical inputs
        shift = data.get("Shift", "Day")
        machine = data.get("Machine_Type", "Type_A")
        material = data.get("Material_Grade", "Standard")
        day = data.get("Day_of_Week", "Monday")

        # Encode categoricals manually
        shift_encoded = [1 if shift == "Evening" else 0,
                         1 if shift == "Night" else 0]

        machine_encoded = [1 if machine == "Type_B" else 0,
                           1 if machine == "Type_C" else 0]

        day_encoded = [
            1 if day == "Monday" else 0,
            1 if day == "Saturday" else 0,
            1 if day == "Sunday" else 0,
            1 if day == "Thursday" else 0,
            1 if day == "Tuesday" else 0,
            1 if day == "Wednesday" else 0
        ]

        material_map = {"Standard": 0, "Premium": 1, "Economy": 2}
        material_encoded = [material_map.get(material, 0)]

        # Combine scaled numeric + encoded categorical features
        scaled_numeric = scaler.transform([inputs])[0]
        final_features = np.array(list(scaled_numeric) + material_encoded + shift_encoded + machine_encoded + day_encoded).reshape(1, -1)

        # Predict
        prediction = model.predict(final_features)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
