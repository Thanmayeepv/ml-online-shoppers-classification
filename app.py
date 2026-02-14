import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Online Shopper Purchase Predictor",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: bold;
    color: #2E4053;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #1F618D;
}
.metric-box {
    background-color: #F4F6F7;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
.footer {
    text-align: center;
    font-size: 14px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🛒 Online Shopper Purchase Prediction</p>', unsafe_allow_html=True)
st.write("Upload your test dataset and evaluate different machine learning models.")

st.markdown("---")

st.sidebar.header("⚙ Model Selection")

model_choice = st.sidebar.selectbox(
    "Select a Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "Naive Bayes",
        "XGBoost"
    ]
)

# Load selected model
model_paths = {
    "Logistic Regression": "model/logistic_model.pkl",
    "Decision Tree": "model/decision_tree_model.pkl",
    "Random Forest": "model/random_forest_model.pkl",
    "KNN": "model/knn_model.pkl",
    "Naive Bayes": "model/naive_bayes_model.pkl",
    "XGBoost": "model/xgboost_model.pkl",
}

model = joblib.load(model_paths[model_choice])
st.sidebar.success(f"{model_choice} model loaded successfully.")

st.markdown('<p class="section-title">📂 Upload Test Dataset (CSV)</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload only test dataset (CSV format)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.dataframe(data.head())

    if "Revenue" not in data.columns:
        st.error("The dataset must contain a 'Revenue' column as the target variable.")
    else:
        X = data.drop("Revenue", axis=1)
        y = data["Revenue"]

        probabilities = model.predict_proba(X)[:, 1]
        
        st.markdown('<p class="section-title">📊 Evaluation Metrics</p>', unsafe_allow_html=True)

        acc = accuracy_score(y, predictions)
        prec = precision_score(y, predictions)
        rec = recall_score(y, predictions)
        f1 = f1_score(y, predictions)
        auc = roc_auc_score(y, probabilities)
        mcc = matthews_corrcoef(y, predictions)
        
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col3.metric("Recall", f"{rec:.4f}")
        
        col4.metric("F1 Score", f"{f1:.4f}")
        col5.metric("AUC Score", f"{auc:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        st.markdown('<p class="section-title">📌 Confusion Matrix</p>', unsafe_allow_html=True)

        cm = confusion_matrix(y, predictions)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted: No", "Predicted: Yes"],
            yticklabels=["Actual: No", "Actual: Yes"],
            ax=ax
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        st.pyplot(fig)

else:
    st.info("Please upload a CSV test dataset to evaluate the selected model.")
