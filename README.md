# Online Shoppers Purchasing Intention - Classification Project

## 1. Problem Statement

The goal of this project is to forecast if the user of an online shopping site will buy a product or not.
The problem is as a binary classification problem, in which the target variable of interest is **Revenue**—where 1 represents **Purchase** and 0 represents **No Purchase**.

---

## 2. Dataset Description

- Source: UCI Machine Learning Repository
- Dataset Name: Online Shoppers Purchasing Intention Dataset
- Number of Instances: 12,330
- Number of Features: 17 input features + 1 target variable
- Target Variable: Revenue (True/False)

The data set appears to include information on the browsing patterns of users, such as visits to pages, the duration of time spent on specific pages, the percentage of visits with high bounce rates, exit rates, the month of the visit, type of visitors, etc.

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

---

## 4. Model Performance Comparison

| Model                | Accuracy | AUC    | Precision | Recall  | F1 Score | MCC    |
|----------------------|----------|--------|-----------|---------|----------|--------|
| Logistic Regression  | 0.8832   | 0.8653 | 0.7640    | 0.3560  | 0.4857   | 0.4696 |
| Decision Tree        | 0.8528   | 0.7290 | 0.5237    | 0.5497  | 0.5364   | 0.4492 |
| K-Nearest Neighbors  | 0.8678   | 0.7888 | 0.6217    | 0.3743  | 0.4673   | 0.4138 |

---

## 5. Observations

- Logistic Regression achieved the highest AUC score, indicating strong probability estimation capability.
- Decision Tree provided better recall compared to other models, meaning it captured more actual positive purchase cases.
- KNN showed balanced but moderate performance.
- The dataset appears to be slightly imbalanced, affecting recall performance in some models.
- Further improvement can be achieved using ensemble models such as Random Forest and XGBoost.
