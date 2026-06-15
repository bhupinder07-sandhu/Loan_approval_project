from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder="../templates")

# Load Models
loan_amount_model = joblib.load("models/loan_amount_model.pkl")
loan_status_model = joblib.load("models/loan_status_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    loan_amount_prediction = None
    loan_status_prediction = None

    if request.method == "POST":

        no_of_dependents = int(request.form["no_of_dependents"])
        education = int(request.form["education"])
        self_employed = int(request.form["self_employed"])
        income_annum = int(request.form["income_annum"])
        loan_term = int(request.form["loan_term"])
        cibil_score = int(request.form["cibil_score"])
        residential_assets_value = int(request.form["residential_assets_value"])
        commercial_assets_value = int(request.form["commercial_assets_value"])
        luxury_assets_value = int(request.form["luxury_assets_value"])
        bank_asset_value = int(request.form["bank_asset_value"])

        amount_features = np.array([[
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]])

        predicted_loan_amount = loan_amount_model.predict(amount_features)[0]

        status_features = np.array([[
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            predicted_loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]])

        status = loan_status_model.predict(status_features)[0]

        loan_amount_prediction = f"₹ {int(predicted_loan_amount):,}"

        if status == 0:
            loan_status_prediction = "Approved ✅"
        else:
            loan_status_prediction = "Rejected ❌"

    return render_template(
        "index.html",
        loan_amount_prediction=loan_amount_prediction,
        loan_status_prediction=loan_status_prediction
    )

app = app