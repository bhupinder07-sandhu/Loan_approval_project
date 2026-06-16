from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(
    __name__,
    template_folder="../templates"
)

# ==========================
# Encoders
# ==========================
le_education = joblib.load("models/le_education.pkl")
le_self_employed = joblib.load("models/le_self_employed.pkl")

# ==========================
# Regression Models
# ==========================
linear_model = joblib.load("models/linear_regression_model.pkl")
dt_regressor = joblib.load("models/decision_tree_regressor_model.pkl")
rf_regressor = joblib.load("models/random_forest_regressor_model.pkl")

# ==========================
# Classification Models
# ==========================
logistic_model = joblib.load("models/logistic_classifier_model.pkl")
dt_classifier = joblib.load("models/decision_tree_classifier_model.pkl")
rf_classifier = joblib.load("models/random_forest_classifier_model.pkl")


# ==========================
# Home Page
# ==========================
@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# Loan Amount Prediction
# ==========================
@app.route("/loan-amount", methods=["GET", "POST"])
def loan_amount():

    prediction = ""

    if request.method == "POST":

        dependents = int(request.form["dependents"])

        education = le_education.transform(
            [request.form["education"]]
        )[0]

        self_employed = le_self_employed.transform(
            [request.form["self_employed"]]
        )[0]

        income_annum = float(request.form["income_annum"])
        loan_term = float(request.form["loan_term"])
        cibil_score = float(request.form["cibil_score"])

        residential_assets_value = float(
            request.form["residential_assets_value"]
        )

        commercial_assets_value = float(
            request.form["commercial_assets_value"]
        )

        luxury_assets_value = float(
            request.form["luxury_assets_value"]
        )

        bank_asset_value = float(
            request.form["bank_asset_value"]
        )

        features = np.array([[
            dependents,
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

        selected_model = request.form["model"]

        if selected_model == "linear":
            model = linear_model

        elif selected_model == "decision_tree":
            model = dt_regressor

        else:
            model = rf_regressor

        result = model.predict(features)[0]

        prediction = f"Predicted Loan Amount: ₹ {result:,.0f}"

    return render_template(
        "loan_amount.html",
        prediction=prediction
    )


# ==========================
# Loan Approval Prediction
# ==========================
@app.route("/loan-approval", methods=["GET", "POST"])
def loan_approval():

    prediction = ""

    if request.method == "POST":

        dependents = int(request.form["dependents"])

        education = le_education.transform(
            [request.form["education"]]
        )[0]

        self_employed = le_self_employed.transform(
            [request.form["self_employed"]]
        )[0]

        income_annum = float(request.form["income_annum"])
        loan_amount = float(request.form["loan_amount"])
        loan_term = float(request.form["loan_term"])
        cibil_score = float(request.form["cibil_score"])

        residential_assets_value = float(
            request.form["residential_assets_value"]
        )

        commercial_assets_value = float(
            request.form["commercial_assets_value"]
        )

        luxury_assets_value = float(
            request.form["luxury_assets_value"]
        )

        bank_asset_value = float(
            request.form["bank_asset_value"]
        )

        features = np.array([[
            dependents,
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]])

        selected_model = request.form["model"]

        if selected_model == "logistic":
            model = logistic_model

        elif selected_model == "decision_tree":
            model = dt_classifier

        else:
            model = rf_classifier

        result = model.predict(features)[0]

        if result == 0:
            prediction = "✅ Loan Approved"
        else:
            prediction = "❌ Loan Rejected"

    return render_template(
        "loan_approval.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)