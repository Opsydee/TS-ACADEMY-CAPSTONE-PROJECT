import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    with open("feature_columns.json", "r") as f:
        features = json.load(f)
    return model, features

model, feature_columns = load_model()

st.title("📡 Customer Churn Prediction")
st.markdown("**Gishon Gold Tech Hub | Group 14** — Supervised Learning & Classification")
st.markdown("---")

st.sidebar.header("Enter Customer Details")
st.sidebar.markdown("Fill in the customer profile below and click **Predict**.")

st.sidebar.subheader("👤 Demographics")
gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen (65+)", ["No", "Yes"])
partner        = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents     = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])

st.sidebar.subheader("📋 Account Information")
tenure            = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract          = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing?", ["Yes", "No"])
payment_method    = st.sidebar.selectbox("Payment Method", [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"])

st.sidebar.subheader("📶 Services Subscribed")
phone_service     = st.sidebar.selectbox("Phone Service?", ["Yes", "No"])
multiple_lines    = st.sidebar.selectbox("Multiple Lines?", ["No", "Yes", "No phone service"])
internet_service  = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security   = st.sidebar.selectbox("Online Security?", ["No", "Yes", "No internet service"])
online_backup     = st.sidebar.selectbox("Online Backup?", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection?", ["No", "Yes", "No internet service"])
tech_support      = st.sidebar.selectbox("Tech Support?", ["No", "Yes", "No internet service"])
streaming_tv      = st.sidebar.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
streaming_movies  = st.sidebar.selectbox("Streaming Movies?", ["No", "Yes", "No internet service"])

st.sidebar.subheader("💰 Financial")
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.25, 118.75, 64.76)
total_charges   = st.sidebar.number_input("Total Charges ($)",
                      min_value=0.0, max_value=9000.0,
                      value=float(tenure * monthly_charges))

predict_button = st.sidebar.button("🔍 Predict Churn", use_container_width=True)

input_data = {
    "gender":            [gender],
    "senior_citizen":    [1 if senior_citizen == "Yes" else 0],
    "partner":           [partner],
    "dependents":        [dependents],
    "tenure":            [tenure],
    "phone_service":     [phone_service],
    "multiple_lines":    [multiple_lines],
    "internet_service":  [internet_service],
    "online_security":   [online_security],
    "online_backup":     [online_backup],
    "device_protection": [device_protection],
    "tech_support":      [tech_support],
    "streaming_tv":      [streaming_tv],
    "streaming_movies":  [streaming_movies],
    "contract":          [contract],
    "paperless_billing": [paperless_billing],
    "payment_method":    [payment_method],
    "monthly_charges":   [monthly_charges],
    "total_charges":     [total_charges],
}

input_df = pd.DataFrame(input_data)

if predict_button:
    prediction    = model.predict(input_df)[0]
    proba         = model.predict_proba(input_df)[0]
    churn_prob    = proba[1] * 100
    no_churn_prob = proba[0] * 100

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("⚠️  HIGH CHURN RISK")
            st.markdown("### This customer is likely to **leave**.")
        else:
            st.success("✅  LOW CHURN RISK")
            st.markdown("### This customer is likely to **stay**.")

        st.markdown("---")
        st.metric(label="Churn Probability",     value=f"{churn_prob:.1f}%")
        st.metric(label="Retention Probability", value=f"{no_churn_prob:.1f}%")

    with col2:
        st.subheader("Probability Breakdown")
        prob_df = pd.DataFrame({
            "Outcome":     ["Will Stay", "Will Churn"],
            "Probability": [no_churn_prob, churn_prob]
        })
        st.bar_chart(prob_df.set_index("Outcome"))

    st.markdown("---")
    st.subheader("Customer Profile Submitted")
    st.dataframe(input_df, use_container_width=True)

    st.markdown("---")
    st.subheader("What Does This Mean?")
    if churn_prob >= 70:
        st.warning("🔴 **Very High Risk**: Immediate retention action recommended. Consider offering a contract upgrade or loyalty discount.")
    elif churn_prob >= 40:
        st.info("🟡 **Moderate Risk**: Monitor this customer closely. Promote add-on services like Online Security or Tech Support.")
    else:
        st.success("🟢 **Low Risk**: Customer appears satisfied. Continue providing consistent service quality.")

else:
    st.info("👈  Fill in the customer details in the sidebar and click **Predict Churn** to get started.")
    st.markdown("### How This Model Works")
    st.markdown("""
    This app uses a **tuned Logistic Regression** model trained on 5,625 customers
    from the IBM Telco Customer Churn dataset.
    - **Accuracy**: 79.8%   |   **F1-Score**: 59.8%
    - **Best Parameters**: C=100, solver=liblinear
    - **Top Churn Drivers**: Month-to-month contract, Fiber optic internet,
      Electronic check payment, High monthly charges
    """)