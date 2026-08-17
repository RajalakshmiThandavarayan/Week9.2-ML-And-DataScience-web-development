from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("models/finalized_model.sav", "rb"))

# Load scaler
scaler = pickle.load(open("models/scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    annual_income = float(request.form["annual_income"])
    financial_strain = 1 if request.form["financial_strain"] == "yes" else 0
    diabetes = 1 if request.form["diabetes"] == "yes" else 0

    #data = np.array([[bgr, bu, sc, pcv, wc]])

    data = pd.DataFrame(
    np.array([[age, annual_income, financial_strain, diabetes]]),
    columns=["age", "annual_income", "financial_strain", "diabetes"])


    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        result = "SDOH Risk Detected"
    else:
        result = "No SDOH Risk Detected"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)