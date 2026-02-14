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
| Naive Bayes          | 0.8439   | 0.8152 | 0.4963    | 0.5236  | 0.5096   | 0.417  |
| Random Forest        | 0.8998   | 0.9178 | 0.732     | 0.5576  | 0.633    | 0.5834 |
| XGBoost              | 0.8796   | 0.9129 | 0.5955    | 0.6437  | 0.6409   | 0.5714 |


---

## 5. Observations

| ML Model Name                 | Observation about model performance                                                                                                                                                                                                                                                                                                                                               |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**       | Achieved high overall accuracy (0.8832) and strong AUC (0.8653), indicating good class separability. However, recall (0.3560) is relatively low compared to precision (0.7640), meaning the model misses many positive cases. The MCC score (0.4696) reflects moderate balanced predictive capability but sensitivity to class imbalance.                                         |
| **Decision Tree**             | Produced moderate accuracy (0.8528) with relatively balanced precision (0.5237) and recall (0.5497). Although it detects more positive cases than Logistic Regression, its lower AUC (0.7290) suggests weaker discrimination ability. The MCC score (0.4492) indicates average classification strength.                                                                           |
| **K-Nearest Neighbors (KNN)** | Delivered good accuracy (0.8678), but recall (0.3743) is limited, showing reduced ability to detect positive instances. Precision (0.6217) is higher than recall, indicating conservative predictions. The MCC score (0.4138) suggests comparatively weaker balanced performance.                                                                                                 |
| **Naive Bayes**               | Achieved accuracy of 0.8439 with relatively balanced precision (0.4963) and recall (0.5236). The AUC score (0.8152) indicates reasonable class discrimination. MCC (0.4170) reflects moderate predictive reliability despite the model’s strong independence assumptions.                                                                                                         |
| **Random Forest (Ensemble)**  | Delivered the highest accuracy (0.8998) and highest AUC (0.9179), demonstrating superior class discrimination. It also achieved the highest MCC (0.5834), indicating strong balanced classification performance. This confirms the advantage of ensemble learning in improving generalization.                                                                                    |
| **XGBoost (Ensemble)**        | Achieved the highest recall (0.6437) and highest F1 score (0.6409), indicating strong detection of positive cases with balanced performance. The MCC (0.5714) is close to Random Forest, showing robust predictive capability. Although accuracy (0.8796) is slightly lower than Random Forest, its higher recall makes it effective when identifying positive cases is critical. |
