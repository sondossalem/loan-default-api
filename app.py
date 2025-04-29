from flask import Flask, request, jsonify
from flask_cors import CORS  
import pickle
import numpy as np
import pandas as pd
import zipfile
import os

# Flask initialization with correct __name__ variable
app = Flask(__name__)
CORS(app)

model_filename = "xgb_pipeline_model.pkl"

# Check if model file exists, otherwise extract it from the zip
if not os.path.exists(model_filename):
    with zipfile.ZipFile("Model.zip", 'r') as zip_ref:
        zip_ref.extractall()

# Load the model from the pickle file
with open(model_filename, "rb") as f:
    model = pickle.load(f)

# Define the columns that the model expects (final columns)
final_columns = [
    'loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'open_acc', 'pub_rec',
    'revol_util', 'mort_acc', 'credit_age', 'loan_issue_year', 'loan_issue_month',
    'sub_grade_A2', 'sub_grade_A3', 'sub_grade_A4', 'sub_grade_A5',
    'sub_grade_B1', 'sub_grade_B2', 'sub_grade_B3', 'sub_grade_B4',
    'sub_grade_B5', 'sub_grade_C1', 'sub_grade_C2', 'sub_grade_C3',
    'sub_grade_C4', 'sub_grade_C5', 'sub_grade_D1', 'sub_grade_D2',
    'sub_grade_D3', 'sub_grade_D4', 'sub_grade_D5', 'sub_grade_E1',
    'sub_grade_E2', 'sub_grade_E3', 'sub_grade_E4', 'sub_grade_E5',
    'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
    'verification_status_Source Verified', 'verification_status_Verified',
    'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational',
    'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase',
    'purpose_medical', 'purpose_moving', 'purpose_other',
    'purpose_renewable_energy', 'purpose_small_business', 'purpose_vacation',
    'purpose_wedding', 'initial_list_status_w', 'application_type_INDIVIDUAL',
    'application_type_JOINT', 'zip_code_05113', 'zip_code_11650',
    'zip_code_22690', 'zip_code_29597', 'zip_code_30723', 'zip_code_48052',
    'zip_code_70466', 'zip_code_86630', 'zip_code_93700'
]

@app.route("/")
def home():
    return "Model API is running with preprocessing and risk analysis!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Step 1: Get data from the POST request
        raw_data = request.get_json()  
        
        # Step 2: Check if all required columns are present
        required_columns = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'home_ownership', 'verification_status', 'purpose', 'issue_d', 'address']
        for column in required_columns:
            if column not in raw_data:
                return jsonify({"error": f"Missing required field: {column}"}), 400

        # Step 3: Convert raw data into DataFrame and preprocess
        df = pd.DataFrame([raw_data])

        # Preprocess the input data
        df['term'] = df['term'].str.extract(r'(\d+)').astype(int)  # Extract term as integer
        df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')  # Replace 'NONE' and 'ANY' with 'OTHER'

        # Handle earliest_cr_line date and calculate credit_age
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
        df['credit_age'] = 2013 - df['earliest_cr_line'].dt.year  # Calculate credit age as the difference from 2013

        # Handle issue_d date and extract loan issue year and month
        df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
        df['loan_issue_year'] = df['issue_d'].dt.year
        df['loan_issue_month'] = df['issue_d'].dt.month

        # Extract zip code from address (last 5 digits)
        df['zip_code'] = df['address'].apply(lambda x: x[-5:])

        # Drop unnecessary columns, including loan_status (target variable)
        drop_cols = ['grade', 'emp_length', 'emp_title', 'title', 'revol_bal', 'pub_rec_bankruptcies',
                     'earliest_cr_line', 'issue_d', 'address', 'loan_status']  # Removed 'loan_status'
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

        # One-hot encode categorical columns
        categorical_cols = ['sub_grade', 'home_ownership', 'verification_status', 'purpose',
                            'initial_list_status', 'application_type', 'zip_code']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

        # Ensure all final columns are present, filling with 0 if missing
        for col in final_columns:
            if col not in df:
                df[col] = 0  # Add missing columns as 0
        df = df[final_columns]  # Keep only the columns expected by the model

        # Step 4: Make the prediction
        prob = model.predict_proba(df)[0][0]  # Get the probability of default
        prediction = int(prob < 0.55)  # Default prediction: 1 if probability < 0.55

        # Step 5: Calculate risk score and risk level
        risk_score = prob * 100  # Calculate risk score as a percentage
        risk_score = round(risk_score, 2)  # Round to two decimal places

        # Determine risk level based on probability
        if prob < 0.3:
            risk_level = "Low Risk"
        elif prob < 0.6:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        # Return prediction and risk information as JSON
        return jsonify({
            "prediction": prediction,
            "risk_score": str(risk_score) + "%",  # Risk score as a percentage string
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default port is 10000, can be changed in environment
    app.run(host="0.0.0.0", port=port)
