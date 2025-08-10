# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

# Load model
@st.cache_resource
def load_model(path=r"H:\ML Project\Customer Churn\data\churn_model.joblib"):
    return joblib.load(path)

# Load test data
@st.cache_data
def load_test_data(path=r"H:\ML Project\Customer Churn\data\test_data.csv"):
    df = pd.read_csv(path)
    return df

model = load_model()
test_df = load_test_data()

# Calculate model accuracy on test data
if 'Churn' in test_df.columns:  # Assuming target column is named 'Churn'
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)
else:
    model_accuracy = None

st.title("📡 Telecom Customer Churn Predictor")
if model_accuracy is not None:
    st.info(f"Model Accuracy on test data: **{model_accuracy:.2%}**")
st.markdown("Enter customer details to get a churn probability and suggestion.")

with st.form("input_form"):
    # Basic info
    gender = st.selectbox("Gender", options=["Male", "Female"])
    senior = st.selectbox("Senior Citizen", options=[0, 1])
    partner = st.selectbox("Partner", options=["Yes", "No"])
    dependents = st.selectbox("Dependents", options=["Yes", "No"])
    
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    
    phone_service = st.selectbox("Phone Service", options=["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", options=["No phone service", "No", "Yes"])
    
    internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", options=["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", options=["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", options=["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", options=["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes", "No internet service"])
    
    contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"])
    payment_method = st.selectbox("Payment Method", 
                                  options=["Bank transfer (automatic)", 
                                           "Credit card (automatic)", 
                                           "Electronic check", 
                                           "Mailed check"])
    
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=1500.0)
    
    submit = st.form_submit_button("Predict")

if submit:
    input_dict = {
        'gender': [gender],
        'SeniorCitizen': [senior],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    }
    
    input_df = pd.DataFrame(input_dict)

    proba = model.predict_proba(input_df)[:, 1][0]
    pred = model.predict(input_df)[0]

    st.metric("Churn probability", f"{proba:.2%}")
    st.write("Predicted class:", "Churn" if pred == 1 else "No churn")
