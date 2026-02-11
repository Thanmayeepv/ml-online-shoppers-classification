# Online Shoppers Purchasing Intention - Classification Project

## 1. Problem Statement

The objective of this project is to predict whether a user browsing an e-commerce website will complete a purchase or not.
This is as a binary classification problem where the target variable is **Revenue** (1 = Purchase, 0 = No Purchase).

---

## 2. Dataset Description

- Source: UCI Machine Learning Repository
- Dataset Name: Online Shoppers Purchasing Intention Dataset
- Number of Instances: 12,330
- Number of Features: 17 input features + 1 target variable
- Target Variable: Revenue (True/False)

The dataset contains information about user browsing behavior such as page visits, duration spent on pages, bounce rates, exit rates, month of visit, visitor type, and other session-level attributes.

---

## 3. Models Implemented

The following classification models are implemented:

1. Logistic Regression
2. Decision Tree Classifi er
3. K-Nearest Neighbor Classifi er
4. Naive Bayes Classifi er - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost

Evaluation Metrics used are as follows:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coeffi cient (MCC Score)
