import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("credit_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("German Credit Risk Prediction App")
st.write("Enter customer details to predict credit risk")

# ---------- CATEGORICAL MAPPINGS ----------

status_dict = {
"No Checking Account":1,
"Balance < 0":2,
"Balance 0 - 200":3,
"Balance > 200":4
}

credit_history_dict = {
"No previous credit / all paid":0,
"All credit paid":1,
"Existing credit paid":2,
"Delay in past payment":3,
"Critical account":4
}

purpose_dict = {
"Car (New)":0,
"Car (Used)":1,
"Furniture/Equipment":2,
"Radio/TV":3,
"Domestic Appliances":4,
"Repairs":5,
"Education":6,
"Vacation":7,
"Retraining":8,
"Business":9,
"Others":10
}

savings_dict = {
"No savings":1,
"<100":2,
"100-500":3,
"500-1000":4,
">=1000":5
}

employment_dict = {
"Unemployed":1,
"<1 year":2,
"1-4 years":3,
"4-7 years":4,
">=7 years":5
}

installment_dict = {
"Low burden":1,
"Moderate":2,
"High":3,
"Very High":4
}

personal_status_dict = {
"Male divorced/separated":1,
"Female divorced/separated":2,
"Male single":3,
"Male married/widowed":4
}

other_debtors_dict = {
"None":1,
"Co-applicant":2,
"Guarantor":3
}

property_dict = {
"Real Estate":1,
"Life Insurance":2,
"Car or Other":3,
"No Property":4
}

other_installment_dict = {
"Bank":1,
"Store":2,
"None":3
}

housing_dict = {
"Rent":1,
"Own":2,
"Free":3
}

job_dict = {
"Unemployed/Unskilled":1,
"Unskilled":2,
"Skilled Worker":3,
"Highly Skilled":4
}

people_liable_dict = {
"1 Person":1,
"2 or More":2
}

telephone_dict = {
"No":1,
"Yes":2
}

foreign_worker_dict = {
"Yes":1,
"No":2
}

# ---------- INPUTS ----------

status = status_dict[st.selectbox("Status", status_dict.keys())]

duration = st.number_input("Loan Duration (months)")

credit_history = credit_history_dict[st.selectbox("Credit History", credit_history_dict.keys())]

purpose = purpose_dict[st.selectbox("Purpose", purpose_dict.keys())]

amount = st.number_input("Credit Amount")

savings = savings_dict[st.selectbox("Savings", savings_dict.keys())]

employment_duration = employment_dict[st.selectbox("Employment Duration", employment_dict.keys())]

installment_rate = installment_dict[st.selectbox("Installment Rate", installment_dict.keys())]

personal_status_sex = personal_status_dict[st.selectbox("Personal Status / Sex", personal_status_dict.keys())]

other_debtors = other_debtors_dict[st.selectbox("Other Debtors", other_debtors_dict.keys())]

present_residence = st.number_input("Present Residence (years)")

property = property_dict[st.selectbox("Property", property_dict.keys())]

age = st.number_input("Age")

other_installment_plans = other_installment_dict[st.selectbox("Other Installment Plans", other_installment_dict.keys())]

housing = housing_dict[st.selectbox("Housing", housing_dict.keys())]

number_credits = st.number_input("Number of Credits")

job = job_dict[st.selectbox("Job", job_dict.keys())]

people_liable = people_liable_dict[st.selectbox("People Liable", people_liable_dict.keys())]

telephone = telephone_dict[st.selectbox("Telephone", telephone_dict.keys())]

foreign_worker = foreign_worker_dict[st.selectbox("Foreign Worker", foreign_worker_dict.keys())]

# ---------- MODEL INPUT ----------

input_data = np.array([[status, duration, credit_history, purpose, amount,
savings, employment_duration, installment_rate,
personal_status_sex, other_debtors, present_residence,
property, age, other_installment_plans, housing,
number_credits, job, people_liable, telephone,
foreign_worker]])

input_scaled = scaler.transform(input_data)

# ---------- PREDICTION ----------

if st.button("Predict Credit Risk"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Good Credit Risk")
    else:
        st.error("Bad Credit Risk")