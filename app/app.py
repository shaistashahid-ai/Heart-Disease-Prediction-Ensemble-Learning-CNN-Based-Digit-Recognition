# app.py — CardioAI Heart Disease Prediction Dashboard
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page setup ─────────────────────────────────────────────
st.set_page_config(page_title="CardioAI Dashboard", page_icon="❤️", layout="wide")

st.title("❤️ CardioAI — Heart Disease Risk Prediction")
st.markdown("*Clinical Decision Support System*")
st.markdown("---")

# ── Load models ─────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(BASE, "xgb_model.json"))

    scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(BASE, "feature_names.pkl"))
    explainer = joblib.load(os.path.join(BASE, "shap_explainer.pkl"))

    with open(os.path.join(BASE, "sample_patient.json")) as f:
        sample = json.load(f)

    return model, scaler, feature_names, explainer, sample


model, scaler, feature_names, explainer, sample = load_artifacts()

CONT_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CAT_COLS = ["cp", "restecg", "slope", "thal"]

# ── SAFE INPUT FUNCTION (FIXED TYPES) ───────────────────────
def num_input(label, key, min_v, max_v, default, step=1, is_float=False):

    if is_float:
        return st.sidebar.number_input(
            label,
            min_value=float(min_v),
            max_value=float(max_v),
            value=float(default),
            step=float(step),
            format="%.1f",
            key=key
        )
    else:
        return st.sidebar.number_input(
            label,
            min_value=int(min_v),
            max_value=int(max_v),
            value=int(default),
            step=int(step),
            format="%d",
            key=key
        )

# ── Sidebar UI ──────────────────────────────────────────────
st.sidebar.header("📋 Patient Data")

age = num_input("Age", "age", 20, 80, sample["age"])

sex = st.sidebar.selectbox(
    "Sex",
    [0, 1],
    index=[0, 1].index(sample["sex"]),
    format_func=lambda x: "Male" if x == 1 else "Female"
)

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3],
    index=[0, 1, 2, 3].index(sample["cp"])
)

trestbps = num_input("Resting BP", "trestbps", 94, 200, sample["trestbps"])
chol = num_input("Cholesterol", "chol", 126, 564, sample["chol"])

fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar",
    [0, 1],
    index=[0, 1].index(sample["fbs"])
)

restecg = st.sidebar.selectbox(
    "Resting ECG",
    [0, 1, 2],
    index=[0, 1, 2].index(sample["restecg"])
)

thalach = num_input("Max Heart Rate", "thalach", 71, 202, sample["thalach"])

exang = st.sidebar.selectbox(
    "Exercise Angina",
    [0, 1],
    index=[0, 1].index(sample["exang"])
)

oldpeak = num_input(
    "ST Depression",
    "oldpeak",
    0,
    6.2,
    sample["oldpeak"],
    step=0.1,
    is_float=True
)

slope = st.sidebar.selectbox(
    "ST Slope",
    [0, 1, 2],
    index=[0, 1, 2].index(sample["slope"])
)

ca = num_input("Major Vessels", "ca", 0, 3, sample["ca"])

thal = st.sidebar.selectbox(
    "Thalassemia",
    [1, 2, 3],
    index=[1, 2, 3].index(sample["thal"])
)

predict_btn = st.sidebar.button("🔍 Predict")

# ── Preprocessing ───────────────────────────────────────────
def preprocess_patient(inputs):
    df = pd.DataFrame([inputs])
    df = pd.get_dummies(df, columns=CAT_COLS)
    df = df.reindex(columns=feature_names, fill_value=0)
    df[CONT_COLS] = scaler.transform(df[CONT_COLS])
    return df

# ── Prediction ──────────────────────────────────────────────
if predict_btn:

    patient = dict(
        age=age, sex=sex, cp=cp,
        trestbps=trestbps, chol=chol,
        fbs=fbs, restecg=restecg,
        thalach=thalach, exang=exang,
        oldpeak=oldpeak, slope=slope,
        ca=ca, thal=thal
    )

    X = preprocess_patient(patient)

    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)

    col1, col2 = st.columns([1, 2])

    with col1:
        if pred:
            st.error(f"🔴 Disease Detected")
        else:
            st.success(f"🟢 No Disease")

        st.metric("Risk Score", f"{proba*100:.1f}%")
        st.progress(float(proba))

    # ── SHAP SAFE HANDLING ───────────────────────────────────
    shap_vals = explainer.shap_values(X)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_vals = shap_vals[0]

    shap_series = pd.Series(np.abs(shap_vals), index=feature_names)
    top3 = shap_series.nlargest(3)

    with col2:
        st.subheader("Feature Impact")

        fig, ax = plt.subplots()

        colors = [
            "red" if shap_vals[feature_names.index(f)] > 0 else "blue"
            for f in top3.index
        ]

        ax.barh(top3.index[::-1], top3.values[::-1], color=colors[::-1])
        ax.set_xlabel("SHAP impact")

        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    st.info(
        f"Top risk factors: {', '.join(top3.index)}"
    )

else:
    st.info("Enter values and click Predict")