import streamlit as st
import pandas as pd
import requests
import json
import os

API_URL = "http://localhost:8001/predict"

st.set_page_config(page_title="Fintech Fraud Dashboard", layout="wide")
st.title("💸 Fintech Fraud Detection Dashboard")

st.markdown("### Test a Transaction for Fraud Risk")
with st.form("tx_form"):
    col1, col2 = st.columns(2)
    amount = col1.number_input("Transaction Amount ($)", min_value=0.01, value=100.0)
    time_since_last_tx = col2.number_input("Time Since Last Transaction (hours)", min_value=0.0, value=2.5)
    account_age_days = col1.number_input("Account Age (days)", min_value=0.0, value=30.0)
    location_risk_score = col2.slider("Location Risk Score", 0.0, 1.0, 0.2)
    
    submitted = st.form_submit_button("Analyze Transaction")
    
if submitted:
    payload = {
        "amount": amount,
        "time_since_last_tx": time_since_last_tx,
        "account_age_days": account_age_days,
        "location_risk_score": location_risk_score
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        prob = result['fraud_probability']
        is_fraud = result['is_fraud']
        
        st.markdown("### Results")
        if is_fraud:
            st.error(f"🚨 FRAUD DETECTED! Risk Score: {prob:.2%}")
        else:
            st.success(f"✅ Transaction Safe. Risk Score: {prob:.2%}")
    except Exception as e:
        st.error(f"Error communicating with API: {e}")
        
if os.path.exists("models/metrics.json"):
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    st.sidebar.markdown("### Model Metrics")
    st.sidebar.metric("F1 Score", f"{metrics['f1_score']:.3f}")
    st.sidebar.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
