import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "iris_model.pkl")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✓ Successfully loaded trained model from {MODEL_PATH}")
else:
    print(f"⚠ Warning: {MODEL_PATH} not found. Please train and save the model first.")

TARGET_NAMES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html", 
            error="Trained model 'iris_model.pkl' could not be found. Please ensure it is present in the project folder."
        )

    # Retrieve form input values
    raw_inputs = {
        "sepal_length": request.form.get("sepal_length", "").strip(),
        "sepal_width": request.form.get("sepal_width", "").strip(),
        "petal_length": request.form.get("petal_length", "").strip(),
        "petal_width": request.form.get("petal_width", "").strip()
    }

    # Validate inputs
    try:
        sl = float(raw_inputs["sepal_length"])
        sw = float(raw_inputs["sepal_width"])
        pl = float(raw_inputs["petal_length"])
        pw = float(raw_inputs["petal_width"])

        # Check realistic biological bounds (0.1 cm to 15.0 cm)
        for name, val in [("Sepal Length", sl), ("Sepal Width", sw), ("Petal Length", pl), ("Petal Width", pw)]:
            if val <= 0:
                raise ValueError(f"{name} must be greater than 0.")
            if val > 15.0:
                raise ValueError(f"{name} value ({val} cm) is unusually large. Please enter a value <= 15.0 cm.")

    except ValueError as e:
        return render_template("index.html", error=str(e), inputs=raw_inputs)

    # Prepare DataFrame input with feature names to match training schema
    sample_df = pd.DataFrame([[sl, sw, pl, pw]], columns=FEATURE_NAMES)

    try:
        # Predict class index
        pred_idx = int(model.predict(sample_df)[0])
        predicted_species = TARGET_NAMES[pred_idx]

        # Calculate prediction confidence
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample_df)[0]
            confidence = round(float(probabilities[pred_idx]) * 100, 1)
        else:
            confidence = 100.0

        return render_template(
            "index.html",
            prediction=predicted_species,
            confidence=confidence,
            inputs=raw_inputs
        )
    except Exception as err:
        return render_template("index.html", error=f"Prediction error: {str(err)}", inputs=raw_inputs)

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Iris Web Application running!")
    print("Access URL: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
