"""
app.py
------
Streamlit web application for the Adult Census Income Salary Predictor.

Run with:
    streamlit run app.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────── paths ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
FEATURE_INFO_PATH = os.path.join(BASE_DIR, "models", "feature_info.pkl")

# ─────────────────────────── load model ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML model…")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURE_INFO_PATH, "rb") as f:
        info = pickle.load(f)
    return model, info


# ─────────────────────────── page config ──────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor — Adult Census",
    page_icon="💼",
    layout="centered",
)

# ─────────────────────────── header ───────────────────────────────────────────
st.title("💼 Salary Prediction App")
st.markdown(
    """
Predict whether an individual's annual income exceeds **\\$50K**
based on census features.  
Fill in the details below and click **Predict**.
"""
)

# ─────────────────────────── check model exists ───────────────────────────────
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_INFO_PATH):
    st.error(
        "⚠️ Trained model not found. "
        "Please run `python train_model.py` first to generate the model files."
    )
    st.stop()

model, info = load_model()

st.sidebar.header("ℹ️ Model Info")
st.sidebar.metric("Best Model", info["best_model_name"])
st.sidebar.metric("Test Accuracy", f"{info['test_accuracy']:.2%}")
st.sidebar.metric("Test ROC-AUC", f"{info['test_roc_auc']:.4f}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** UCI Adult Census Income  \n**Target:** `income` (<=50K / >50K)")

# ─────────────────────────── input form ───────────────────────────────────────
st.markdown("---")
st.subheader("📋 Enter Individual Details")

nr = info["num_ranges"]
cu = info["cat_unique"]

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=int(nr["age"]["min"]),
        max_value=int(nr["age"]["max"]),
        value=int(nr["age"]["median"]),
        step=1,
    )
    educational_num = st.slider(
        "Education Level (numeric)",
        min_value=int(nr["educational-num"]["min"]),
        max_value=int(nr["educational-num"]["max"]),
        value=int(nr["educational-num"]["median"]),
        help="1 = Preschool … 16 = Doctorate",
    )
    hours_per_week = st.slider(
        "Hours per Week",
        min_value=int(nr["hours-per-week"]["min"]),
        max_value=int(nr["hours-per-week"]["max"]),
        value=int(nr["hours-per-week"]["median"]),
    )
    capital_gain = st.number_input(
        "Capital Gain ($)",
        min_value=int(nr["capital-gain"]["min"]),
        max_value=int(nr["capital-gain"]["max"]),
        value=0,
        step=100,
    )
    capital_loss = st.number_input(
        "Capital Loss ($)",
        min_value=int(nr["capital-loss"]["min"]),
        max_value=int(nr["capital-loss"]["max"]),
        value=0,
        step=100,
    )
    fnlwgt = st.number_input(
        "Final Weight (fnlwgt)",
        min_value=int(nr["fnlwgt"]["min"]),
        max_value=int(nr["fnlwgt"]["max"]),
        value=int(nr["fnlwgt"]["median"]),
        step=1000,
        help="Census sampling weight — leave at default if unsure.",
    )

with col2:
    workclass = st.selectbox("Workclass", options=cu["workclass"])
    education = st.selectbox("Education", options=cu["education"])
    marital_status = st.selectbox("Marital Status", options=cu["marital-status"])
    occupation = st.selectbox("Occupation", options=cu["occupation"])
    relationship = st.selectbox("Relationship", options=cu["relationship"])
    race = st.selectbox("Race", options=cu["race"])
    gender = st.selectbox("Gender", options=cu["gender"])
    native_country = st.selectbox("Native Country", options=cu["native-country"])

# ─────────────────────────── predict ──────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔮 Predict Income", use_container_width=True, type="primary")

if predict_btn:
    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "fnlwgt": fnlwgt,
                "educational-num": educational_num,
                "capital-gain": capital_gain,
                "capital-loss": capital_loss,
                "hours-per-week": hours_per_week,
                "workclass": workclass,
                "education": education,
                "marital-status": marital_status,
                "occupation": occupation,
                "relationship": relationship,
                "race": race,
                "gender": gender,
                "native-country": native_country,
            }
        ]
    )

    # Ensure column order matches training
    input_data = input_data[info["all_features"]]

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    label_classes = info["label_encoder_classes"]  # ['<=50K', '>50K']
    predicted_label = label_classes[prediction]
    confidence = prob[prediction]

    st.markdown("### 🎯 Prediction Result")

    if predicted_label == ">50K":
        st.success(
            f"**Predicted Income: {predicted_label}** — This individual is likely to earn **more than $50,000** per year."
        )
    else:
        st.warning(
            f"**Predicted Income: {predicted_label}** — This individual is likely to earn **$50,000 or less** per year."
        )

    c1, c2 = st.columns(2)
    c1.metric("Prediction", predicted_label)
    c2.metric("Confidence", f"{confidence:.1%}")

    # Probability bar
    st.markdown("#### Prediction Probabilities")
    prob_df = pd.DataFrame(
        {"Income Class": label_classes, "Probability": [f"{p:.1%}" for p in prob]}
    )
    st.bar_chart(
        data=pd.DataFrame({"Probability": prob}, index=label_classes),
        use_container_width=True,
    )

    with st.expander("📊 Input Summary"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}))

# ─────────────────────────── footer ───────────────────────────────────────────
st.markdown("---")
st.caption(
    "Adult Census Income dataset · UCI Machine Learning Repository · "
    "Model: scikit-learn Pipeline"
)
