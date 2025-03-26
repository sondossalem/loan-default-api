from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import zipfile
import os

app = Flask(__name__)

# اسم ملف الموديل داخل الملف المضغوط
model_filename = "xgb_pipeline_model.pkl"

# فك ضغط Model.zip لاستخراج الموديل إذا لم يكن موجودًا
if not os.path.exists(model_filename):
    with zipfile.ZipFile("Model.zip", 'r') as zip_ref:
        zip_ref.extractall()

# تحميل الموديل المدرب
with open(model_filename, "rb") as f:
    model = pickle.load(f)

# الأعمدة التي يتوقعها الموديل بالتحديد
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

        # تحقق من وجود جميع الأعمدة المطلوبة
        for col in expected_columns:
            if col not in input_df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400

        # ترتيب الأعمدة حسب ما يتوقعه الموديل
        input_df = input_df[expected_columns]

        # تنفيذ التنبؤ
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
