from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ تفعيل CORS
import pickle
import numpy as np
import pandas as pd
import zipfile
import os

app = Flask(__name__)
CORS(app)  # ✅ تفعيل CORS للسماح للواجهة تتواصل

# اسم ملف الموديل داخل الملف المضغوط
model_filename = "xgb_pipeline_model.pkl"

# فك ضغط Model.zip لاستخراج الموديل إذا لم يكن موجودًا
if not os.path.exists(model_filename):
    with zipfile.ZipFile("Model.zip", 'r') as zip_ref:
        zip_ref.extractall()

with open(model_filename, "rb") as f:
    model = pickle.load(f)

# الأعمدة التي يتوقعها الموديل بالتحديد
expected_columns = [
    "loan_amnt", "term", "int_rate", "sub_grade", "home_ownership", "annual_inc",
    "verification_status", "purpose", "dti", "open_acc", "pub_rec", "revol_util",
    "initial_list_status", "application_type", "mort_acc", "loan_issue_year",
    "loan_issue_month", "credit_age", "zip_code"
]

# الأعمدة التي تم ترميزها خلال التدريب (بعد get_dummies)
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
    'verification_status_Not Verified',
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

        # معالجة الأعمدة المدخلة
        input_df['term'] = input_df['term'].str.extract(r'(\d+)').astype(int)
        input_df['home_ownership'] = input_df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

        # معالجة التواريخ المدخلة بتنسيق "Month YYYY"
        input_df['earliest_cr_line'] = pd.to_datetime(input_df['earliest_cr_line'], format='%B %Y', errors='coerce')
        input_df['issue_d'] = pd.to_datetime(input_df['issue_d'], format='%B %Y', errors='coerce')

        # التحقق من وجود NaT في التواريخ
        if input_df['earliest_cr_line'].isnull().any() or input_df['issue_d'].isnull().any():
            return jsonify({"error": "Invalid date format for 'earliest_cr_line' or 'issue_d'"}), 400

        # إضافة الأعمدة الفئوية باستخدام get_dummies
        categorical_cols = ['sub_grade', 'home_ownership', 'verification_status', 'purpose', 
                            'initial_list_status', 'application_type', 'zip_code']
        input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

        # التأكد من أن جميع الأعمدة الموجودة في final_columns موجودة
        for col in final_columns:
            if col not in input_df:
                input_df[col] = 0
        input_df = input_df[final_columns]

        # التنبؤ بحساب الاحتمال
        prob = model.predict_proba(input_df)[0][0]

        # التحقق من وجود NaN في الاحتمال
        if np.isnan(prob):
            return jsonify({"error": "Invalid prediction result!"}), 400

        prediction = int(prob < 0.55)  # 1 = Fully Paid إذا احتمال التعثر منخفض

        # تحديد مستوى المخاطرة
        if prob < 0.3:
            risk_level = "Low Risk"
        elif prob < 0.6:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        # تحويل الـ risk_score إلى نسبة مئوية
        risk_score = round(prob * 100, 2)

        return jsonify({
            "prediction": prediction,
            "risk_score": f"{risk_score}%",
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
