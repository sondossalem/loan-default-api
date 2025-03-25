
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the expected input columns
expected_columns = [
    "loan_amnt", "term", "int_rate", "sub_grade", "home_ownership", "annual_inc",
    "verification_status", "purpose", "dti", "open_acc", "pub_rec", "revol_util",
    "initial_list_status", "application_type", "mort_acc", "loan_issue_year",
    "loan_issue_month", "credit_age", "zip_code"
]

@app.route("/")
def home():
    return "Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Ensure all expected columns are present
        for col in expected_columns:
            if col not in input_df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400

        # Predict
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
