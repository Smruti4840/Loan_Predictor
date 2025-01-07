import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the trained model
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title("Loan Prediction Web Application")

# Input fields for the user
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Married", "Unmarried"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self-Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount", min_value=0, value=100)
loan_term = st.selectbox("Loan Term (years)", [15, 20, 30])
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Submit button
if st.button("Predict Loan Approval"):
    # Prepare input data
    input_data = {
        'gender': [gender],
        'married': [married],
        'dependents': [dependents],
        'education': [education],
        'self_employed': [self_employed],
        'applicant_income': [applicant_income],
        'coapplicant_income': [coapplicant_income],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term * 12],  # Convert loan term to months
        'credit_history': [credit_history],
        'property_area': [property_area]
    }

    input_df = pd.DataFrame(input_data)

    # Encoding categorical columns using LabelEncoder
    label_encoder = LabelEncoder()

    input_df['gender'] = label_encoder.fit_transform(input_df['gender'])
    input_df['married'] = label_encoder.fit_transform(input_df['married'])
    input_df['dependents'] = label_encoder.fit_transform(input_df['dependents'])
    input_df['education'] = label_encoder.fit_transform(input_df['education'])
    input_df['self_employed'] = label_encoder.fit_transform(input_df['self_employed'])
    input_df['property_area'] = label_encoder.fit_transform(input_df['property_area'])

    # Handle any missing values (if any)
    input_df.fillna(input_df.median(), inplace=True)

    # Create 'total_income' and 'loan_income_ratio' columns (the missing features)
    input_df['total_income'] = input_df['applicant_income'] + input_df['coapplicant_income']
    input_df['loan_income_ratio'] = input_df['loan_amount'] / input_df['total_income']

    # Display input data for confirmation
    st.subheader("Input Data:")
    st.write(input_df)

    # Check predicted probabilities
    probability = model.predict_proba(input_df)
    st.write("Prediction Probability:", probability)

    # Apply a threshold for classification
    threshold = 0.6
    if probability[0][1] > threshold:
        prediction = 1  # Approved
        st.success("The loan is likely to be approved!")
    else:
        prediction = 0  # Rejected
        st.error("The loan is likely to be rejected.")
