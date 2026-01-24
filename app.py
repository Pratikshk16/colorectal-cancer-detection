from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained artifacts
model = joblib.load("artifacts/model/model.pkl")
preprocessor = joblib.load("artifacts/processed/preprocessor.pkl")


def get_top_factors(model, transformed_input, feature_names, top_n=3):
    coefs = model.coef_[0]
    contribution = coefs * transformed_input[0]

    feature_contrib = list(zip(feature_names, contribution))
    feature_contrib.sort(key=lambda x: abs(x[1]), reverse=True)

    readable = []
    for name, _ in feature_contrib[:top_n]:
        clean = name.replace("cat__", "").replace("num__", "")
        readable.append(clean)

    return readable


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Numeric
        age = int(request.form["Age"])
        tumor_size = float(request.form["Tumor_Size_mm"])

        # Categorical (strings)
        cancer_stage = request.form["Cancer_Stage"]
        treatment_type = request.form["Treatment_Type"]
        early_detection = request.form["Early_Detection"]
        screening_history = request.form["Screening_History"]

        input_df = pd.DataFrame([{
            "Age": age,
            "Tumor_Size_mm": tumor_size,
            "Cancer_Stage": cancer_stage,
            "Treatment_Type": treatment_type,
            "Early_Detection": early_detection,
            "Screening_History": screening_history
        }])

        transformed = preprocessor.transform(input_df)

        prob = model.predict_proba(transformed)[0][1]
        prediction = "Yes" if prob >= 0.35 else "No"

        feature_names = preprocessor.get_feature_names_out()
        factors = get_top_factors(model, transformed, feature_names)

        return render_template(
            "index.html",
            prediction=prediction,
            probability=round(prob * 100, 2),
            factors=factors
        )

    except Exception as e:
        return render_template("index.html", prediction="Error", probability=str(e))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
