import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("xgboost_model.pkl")

st.title("Online Shopper Purchase Prediction")
st.write("Predict whether a user will complete a purchase.")

st.header("Enter Session Details")

# ---------------- NUMERIC INPUTS ----------------
Administrative = st.number_input("Administrative Pages", min_value=0)
Administrative_Duration = st.number_input("Administrative Duration", min_value=0.0)
Informational = st.number_input("Informational Pages", min_value=0)
Informational_Duration = st.number_input("Informational Duration", min_value=0.0)
ProductRelated = st.number_input("Product Related Pages", min_value=0)
ProductRelated_Duration = st.number_input("Product Related Duration", min_value=0.0)
BounceRates = st.number_input("Bounce Rates", min_value=0.0, max_value=1.0)
ExitRates = st.number_input("Exit Rates", min_value=0.0, max_value=1.0)
PageValues = st.number_input("Page Values", min_value=0.0)
SpecialDay = st.number_input("Special Day", min_value=0.0, max_value=1.0)

OperatingSystems = st.number_input("Operating Systems", min_value=1)
Browser = st.number_input("Browser", min_value=1)
Region = st.number_input("Region", min_value=1)
TrafficType = st.number_input("Traffic Type", min_value=1)

# ---------------- CATEGORICAL INPUTS ----------------
month_options = ["Feb","Mar","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]
Month_input = st.selectbox("Month", month_options)

visitor_options = ["New_Visitor", "Returning_Visitor", "Other"]
Visitor_input = st.selectbox("Visitor Type", visitor_options)

Weekend_input = st.selectbox("Weekend", ["No", "Yes"])

# ---------------- MANUAL ENCODING ----------------
month_mapping = {
    "Feb": 0,
    "Mar": 1,
    "May": 2,
    "June": 3,
    "Jul": 4,
    "Aug": 5,
    "Sep": 6,
    "Oct": 7,
    "Nov": 8,
    "Dec": 9
}

visitor_mapping = {
    "New_Visitor": 0,
    "Returning_Visitor": 1,
    "Other": 2
}

Month = month_mapping[Month_input]
VisitorType = visitor_mapping[Visitor_input]
Weekend = 1 if Weekend_input == "Yes" else 0

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    input_data = np.array([[Administrative,
                            Administrative_Duration,
                            Informational,
                            Informational_Duration,
                            ProductRelated,
                            ProductRelated_Duration,
                            BounceRates,
                            ExitRates,
                            PageValues,
                            SpecialDay,
                            Month,
                            OperatingSystems,
                            Browser,
                            Region,
                            TrafficType,
                            VisitorType,
                            Weekend]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success(f" Likely to complete purchase (Probability: {probability:.2f})")
    else:
        st.error(f" Unlikely to complete purchase (Probability: {probability:.2f})")
