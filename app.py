from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ تفعيل CORS
import pickle
import numpy as np
import pandas as pd
import zipfile
import os

app = Flask(__name__)
CORS(app)  # ✅ تفعيل CORS للسماح للواجهة تتواصل

model_filename = "xgb_pipeline_model.pkl"

if not os.path.exists(model_filename):
    with zipfile.ZipFile("Model.zip", 'r') as zip_ref:
        zip_ref.extractall()

with open(model_filename, "rb") as f:
    model = pickle.load(f)

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
        # استلام البيانات المدخلة
        raw_data = request.get_json()
        df = pd.DataFrame([raw_data])

        # فحص القيم المفقودة في البيانات
        if df.isnull().sum().any():
            return jsonify({"error": "Missing data in required fields!"}), 400

        # تحويل 'term' إلى رقم باستخدام str.extract
        df['term'] = df['term'].str.extract(r'(\d+)').astype(int)

        # تحويل 'earliest_cr_line' إلى تاريخ واستخراج 'credit_age'
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
        df['credit_age'] = 2013 - df['earliest_cr_line'].dt.year  # حساب العمر الائتماني بناءً على السنة

        # استخراج 'zip_code' من 'address' (آخر 5 أرقام)
        df['zip_code'] = df['address'].apply(lambda x: x[-5:])

        # باقي العمليات على الأعمدة
        drop_cols = ['grade', 'emp_length', 'emp_title', 'title', 'revol_bal', 'pub_rec_bankruptcies',
                     'earliest_cr_line', 'address']
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

        # تحويل الأعمدة الفئوية
        categorical_cols = ['sub_grade', 'home_ownership', 'verification_status', 'purpose',
                            'initial_list_status', 'application_type', 'zip_code']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # التأكد من أن جميع الأعمدة موجودة
        for col in final_columns:
            if col not in df:
                df[col] = 0
        df = df[final_columns]

        # حساب الاحتمال
        prob = model.predict_proba(df)[0][0]

        # التحقق من وجود NaN في الاحتمال
        if np.isnan(prob):
            return jsonify({"error": "Invalid prediction result!"}), 400

        prediction = int(prob < 0.55)

        # تحديد مستوى المخاطرة بناءً على الاحتمال
        if prob < 0.3:
            risk_level = "Low Risk"
        elif prob < 0.6:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        # تحويل الـ risk_score إلى نسبة مئوية
        risk_score = round(prob * 100, 2)  # ضرب في 100 لتحويلها إلى نسبة مئوية

        return jsonify({
            "prediction": prediction,
            "risk_score": f"{risk_score}%",  # إضافة الـ % بعد النسبة المئوية
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
