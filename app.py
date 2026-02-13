import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Online Shoppers Purchase Prediction")

st.title("Online Shoppers Purchasing Intention Prediction")

# -----------------------------
# Model Selection
# -----------------------------
st.sidebar.header("Model Selection")

model_option = st.sidebar.selectbox(
    "Choose a Model",
    ("XGBoost", "Logistic Regression", "Decision Tree", "KNN","Naive Bayes")
)

if model_option == "XGBoost":
    model = joblib.load("model/xgboost_model.pkl")
elif model_option == "Logistic Regression":
    model = joblib.load("model/logistic_model.pkl")
elif model_option == "Decision Tree":
    model = joblib.load("model/decision_tree_model.pkl")
elif model_option == "Naive Bayes":
    model = joblib.load("model/naive_bayes_model.pkl")
elif model_option == "Random Forest":
    model = joblib.load("model/Random_Forest_model.pkl")
else:
    model = joblib.load("model/knn_model.pkl")

st.sidebar.success(f"{model_option} Loaded Successfully")

# -----------------------------
# Dataset Upload
# -----------------------------
st.header("Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(data.head())

    if "Revenue" not in data.columns:
        st.error("CSV must contain 'Revenue' column as target.")
    else:
        X = data.drop("Revenue", axis=1)
        y = data["Revenue"]

        y_pred = model.predict(X)

        # -----------------------------
        # Evaluation Metrics
        # -----------------------------
        st.header("Model Evaluation Metrics")

        accuracy = accuracy_score(y, y_pred)
        st.write(f"**Accuracy:** {accuracy:.4f}")

        st.subheader("Classification Report")
        report = classification_report(y, y_pred)
        st.text(report)

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

else:
    st.info("Please upload a test CSV file to evaluate the model.")
