# :point_right: Machine Learning-Based Employee Attrition Prediction and Risk Scoring System
### A Machine Learning-Based system built for HR executives to predict the attrition risk for an employee.
---
# 📊 Model Evaluation Performance

| Model Variant | 🎯 Accuracy | 📍 Precision | 🔍 Recall | ⚖️ F1 Score | 📈 AUC Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** (LogisticRegression) | `74.49%` | `34.44%` | **`65.96%`** | **`45.26%`** | `78.80%` |
| **Selected** (RandomForestClassifier) | **`84.69%`** | **`58.33%`** | `14.89%` | `23.73%` | **`80.80%`** |

---
## 📌 Executive Summary & Problem Statement

### 🔍 The Challenge
In operational environments, sudden employee attrition acts exactly like an unpredicted machine failure—causing immediate production bottlenecks, skyrocketing turnover costs, and critical safety gaps. Traditional HR workflows are entirely reactive, identifying burnout only after a resignation letter is submitted. This project frames workforce management as a **predictive maintenance** problem, treating high-value talent as a critical system layer where hidden telemetry signals must be monitored to predict and prevent sudden operational failures.

### ⚙️ The Solution
To solve this, I developed an end-to-end machine learning pipeline that transforms historical employee metrics into actionable risk logs. The repository evaluates a **Logistic Regression** baseline against an optimized **Random Forest Classifier**, utilizing a **Calibrated Classifier (`CalibratedClassifierCV`)** layer. This structural calibration ensures that the model outputs true, reliable probability risk percentages rather than binary labels, giving operations managers exact timelines for early intervention.

### 🚀 High-Level Impact
* **Engineered Precision:** The final Random Forest model delivers an **84.69% Classification Accuracy** and an **80.80% AUC Score**, ensuring reliable risk stratification.
* **Live Operational Dashboard:** The entire pipeline is deployed as a live, interactive web tool on **Streamlit Cloud**, allowing operation headers to upload staff logs, visually isolate high-risk teams, and schedule preemptive retention workflows before an operational disruption occurs.

