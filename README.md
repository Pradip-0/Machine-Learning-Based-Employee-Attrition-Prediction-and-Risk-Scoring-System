# :point_right: Machine Learning-Based Employee Attrition Prediction and Risk Scoring System
### A Machine Learning-Based system built for HR executives to predict the attrition risk for an employee.


---
## 📌 Executive Summary & Problem Statement

### 🔍 The Challenge
In operational environments, sudden employee attrition acts exactly like an unpredicted machine failure—causing immediate production bottlenecks, skyrocketing turnover costs, and critical safety gaps. Traditional HR workflows are entirely reactive, identifying burnout only after a resignation letter is submitted. This project frames workforce management as a **predictive maintenance** problem, treating high-value talent as a critical system layer where hidden telemetry signals must be monitored to predict and prevent sudden operational failures.

### ⚙️ The Solution
To solve this, I developed an end-to-end machine learning pipeline that transforms historical employee metrics into actionable risk logs. The repository evaluates a **Logistic Regression** baseline against an optimized **Random Forest Classifier**, utilizing a **Calibrated Classifier (`CalibratedClassifierCV`)** layer. This structural calibration ensures that the model outputs true, reliable probability risk percentages rather than binary labels, giving operations managers exact timelines for early intervention.

### 🚀 High-Level Impact
* **Engineered Precision:** The final Random Forest model delivers an **84.69% Classification Accuracy** and an **80.80% AUC Score**, ensuring reliable risk stratification.
* **Live Operational Dashboard:** The entire pipeline is deployed as a live, interactive web tool on **Streamlit Cloud**, allowing operation headers to upload staff logs, visually isolate high-risk teams, and schedule preemptive retention workflows before an operational disruption occurs.

---
## 🛠️ System Architecture & Workflow:

### 🛠️ Data Ingestion
We sourced the foundational attrition dataset directly from Palo Alto Networks in CSV format, ensuring a robust, real-world data baseline for our analysis.

### 📋 Data Dictionary & Feature Schema

The dataset consists of 30 employee-centric attributes, encompassing demographic details, compensation structures, role-specific metrics, and engagement scores. The target variable is historical employee attrition.

#### 🎯 Target Variable
*   **`Attrition`**: Indicates whether the employee left the company (`1` = Left, `0` = Retained).

#### 👤 Employee Demographics & Background
*   **`Age`**: Current age of the employee.
*   **`Gender`**: Gender identity (`Male`, `Female`).
*   **`MaritalStatus`**: Current marital status (`Single`, `Married`, `Divorced`).
*   **`Education`**: Formal education level, mapped categorically from `1` (Lowest) to `5` (Highest).
*   **`EducationField`**: Primary field of study and educational background.

#### 💼 Role & Work Environment
*   **`Department`**: Organizational department where the employee is assigned.
*   **`JobRole`**: Specific professional job title.
*   **`JobLevel`**: Job seniority tier, ranked from `1` (Entry-level) to `5` (Executive).
*   **`BusinessTravel`**: Frequency of work-related travel (`Non-Travel`, `Travel_Rarely`, `Travel_Frequently`).
*   **`DistanceFromHome`**: Commute distance between the employee's residence and the office (measured in miles).
*   **`OverTime`**: Indicates if the employee regularly works beyond standard hours (`Yes`, `No`).

#### 💰 Compensation & Financials
*   **`MonthlyIncome`**: Gross monthly take-home salary.
*   **`DailyRate`**: Calculated daily compensation rate.
*   **`HourlyRate`**: Contractual hourly pay rate.
*   **`MonthlyRate`**: Calculated monthly compensation rate benchmark.
*   **`PercentSalaryHike`**: Percentage increase in salary received during the last appraisal cycle.
*   **`StockOptionLevel`**: Number of corporate stock options allocated, scaled from `0` to `3`.

#### 📈 Sentiment & Engagement Metrics
*   **`JobInvolvement`**: Level of active engagement in their role, scored from `1` (Low) to `4` (High).
*   **`JobSatisfaction`**: Self-reported job satisfaction score, scaled from `1` (Low) to `4` (High).
*   **`RelationshipSatisfaction`**: Peer and workplace relationship quality score, scaled from `1` (Low) to `4` (High).
*   **`WorkLifeBalance`**: Perceived equilibrium between work and personal life, scored from `1` (Bad) to `4` (Best).
*   **`PerformanceRating`**: Most recent formal performance evaluation score.

#### ⏳ Tenure & Career History
*   **`TotalWorkingYears`**: Total cumulative professional experience across all companies.
*   **`NumCompaniesWorked`**: Total number of distinct organizations the employee has been employed by.
*   **`YearsAtCompany`**: Continuous tenure length at the current organization.
*   **`YearsInCurrentRole`**: Total time spent in the employee's current job role.
*   **`YearsSinceLastPromotion`**: Years elapsed since the employee's most recent internal promotion.
*   **`YearsWithCurrManager`**: Tenure duration under the reporting line of the current manager.
*   **`TrainingTimesLastYear`**: Number of professional development training courses completed in the previous year.

### 📊 Exploratory Data Analysis (EDA)
Before writing any modeling code, we conducted a thorough EDA phase to unlock key behavioral insights. Our primary objectives were twofold:
* **Pinpoint Attrition Patterns:** Identifying specific employee segments exhibiting the highest risk of leaving.
* **Isolate Risk Drivers:** Uncovering the underlying features and organizational factors actively pushing employees toward attrition.

### ⚙️ Preprocessing & Feature Engineering
Data preparation was executed through a rigorous pipeline to ensure high data quality while preserving critical signals:
* **Sanitization:** Standardized cleaning routines were applied to detect duplicate records, handle missing values, and isolate statistical outliers.
* **Strategic Outlier Retention:** Because attrition data is naturally sparse, we intentionally retained outliers to prevent worsening the class imbalance problem.
* **Feature Creation:** Engineered targeted domain-specific features driven directly by our EDA findings to boost model predictive power.
* **Imbalance Mitigation:** To address the severe class skew, we applied **Adaptive Synthetic Sampling (ADASYN)** via the `imblearn` library. This shifted the decision boundary focus toward harder-to-learn minority examples.
* **Artifact Storage:** The resulting balanced dataset was versioned and saved as optimized CSV files, creating a clean checkpoint for downstream modeling.

### 🚀 Training Pipeline
Our modeling strategy focused on benchmarking advanced ensemble methods against traditional baselines:
* **Model Selection:** We evaluated four distinct classifiers: `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, and `XGBClassifier`. 
* **Baseline Benchmarking:** `LogisticRegression` was established as our control baseline to measure the net performance gain of our ensemble architectures.
* **Automated Hyperparameter Tuning:** Instead of brute-force grid searches, we implemented **Bayesian Optimization** using `optuna` to efficiently navigate the hyperparameter space for each model.
* **Multi-Metric Evaluation:** Final model selection was not based on accuracy alone. We thoroughly cross-validated and ranked the models across 5 key performance indicators: **Accuracy**, **Precision**, **Recall**, **F1-score**, and **AUC-ROC** to ensure a balanced, robust deployment candidate.


---
## 🚀 Model Selection and Interpretability Pipeline

| Model Variant | 🎯 Accuracy | 📍 Precision | 🔍 Recall | ⚖️ F1 Score | 📈 AUC Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| *LogisticRegression* (**Baseline**) | `74.49%` | `34.44%` | **`65.96%`** | **`45.26%`** | `78.80%` |
| *RandomForestClassifier* (**Selected**) | **`84.69%`** | **`58.33%`** | `14.89%` | `23.73%` | **`80.80%`** |
| *GradientBoostingClassifier* | `85.37%` | `66.67%` | `17.02%` | `27.12%` | `78.30%` |
| *XGBClassifier* | `87.07%` | `73.68%` | `29.79%` | `42.42%` | `76.78%` |

### ⚠️ Key Limitations & Final Model Selection

*   **Synthetic Data Bottleneck:** Due to severe initial class imbalance, generating a high volume of synthetic data via ADASYN for the minority class (`Attrition = 1`) caused a noticeable drop in recall across all tree-based ensemble models on real-world validation data. The synthetic samples failed to capture the nuances of real-world attrition fully. This remains a critical project limitation that will be addressed in future iterations by exploring alternative architectures (e.g., cost-sensitive neural networks or anomaly detection frameworks).
*   **AUC-Driven Model Selection:** To balance performance despite this bottleneck, the **`RandomForestClassifier`** was selected as our final production model due to its superior and stable AUC-ROC score, which demonstrates its strong overall discriminative power.
*   **Probability Calibration:** To ensure production readiness, the final model was wrapped with **`CalibratedClassifierCV`** for two critical reasons:
    *   **Fixes Biased Probabilities:** Random Forest averages independent tree votes, structurally pushing probability outputs away from 0 and 1 toward a conservative center; calibration corrects this distortion to reflect true real-world risk percentages.
    *   **Enables Accurate HR Action:** It guarantees that an output score of 0.80 correlates to an exact 80% empirical chance of employee attrition, empowering leadership with highly reliable risk prioritization and financial forecasting tools.

### 🔍 Feature Importance & Model Explainability

To ensure our attrition predictions are completely transparent and actionable for HR stakeholders, we implemented a multi-layered explainability framework rather than relying on a single metric:

*   **Beyond Surface-Level Metrics:** We initially generated a standard **Feature Importance Graph** from our calibrated Random Forest model. However, global feature importance alone cannot capture the complete picture, as it highlights *that* a feature matters, but fails to show *how* or *why*.
*   **Deep-Dive with SHAP Analysis:** To uncover the exact directional impact of each variable, we conducted a **SHAP (Shapley Additive explanations) value analysis**. This game-theoretic approach isolated the top 10 most influential features driving employee attrition, revealing both global trends and individual sample dynamics.
*   **Mapping Non-Linear Trends:** To visualize exactly how specific numeric ranges affect risk, we plotted **Partial Dependence Plots (PDPs)** for these top 10 features. This step explicitly showed us the tipping points where risk levels jump (e.g., the exact distance from home in miles where attrition probability spikes).
*   **Actionable Counterfactual Explanations:** Finally, we built a custom programmatic function to deliver **Counterfactual Explanations** (What-If analysis). This tool provides HR managers with prescriptive, real-world solutions by showing the minimum organizational changes required to flip a high-risk employee back to a retention state (e.g., *"If this employee's Overtime is changed to 'No' and Monthly Income is increased by $400, their predicted attrition risk drops below 30%"*).

---
## 🚀 Model Deployment & Distribution Strategy

The final calibrated model and interpretability tools are packaged into an interactive **Streamlit Web Application** deployed via **Streamlit Cloud**. To maximize enterprise privacy and data governance, we have structured this application for decentralized deployment:

*   **Production Host:** The reference application is live on Streamlit Cloud, serving as an immediate visual proof-of-concept for interactive attrition analysis.
*   **Decentralized Deployment Architecture:** To ensure sensitive employee data remains completely isolated and compliant with internal data security protocols, direct public access to our hosted app is not recommended for production workloads. 
*   **Seamless On-Premises Cloning:** Stakeholders can easily fork or clone the source repository, including all serialized model artifacts, configurations, and directory structures. The application can then be spun up instantly in a private enterprise Streamlit Cloud account or local environment via a simple terminal command:
    ```bash
    streamlit run app.py
    ```

## ⚙️ Installation & Environment Setup:
This project was natively developed within a **Kaggle Notebook environment**. To replicate the workspace locally or migrate it to an enterprise infrastructure, follow the configuration steps below.

### 🖥️ Hardware & Environment Requirements
*   **Compute:** CPU-only execution. No dedicated GPU or CUDA configuration is required for training, optimization, or inference.
*   **Path Configurations:** Ensure you update all relative file paths and environment variables in the source scripts to match your local directory structure instead of the default Kaggle pathing (`/kaggle/input/...`).

### 🐍 Virtual Environment Setup
To prevent dependency conflicts with your global Python installation, isolate the project using a virtual environment:

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder-name>

# Create a virtual environment (Python 3.9+ recommended)
python -bin venv attrition_env

# Activate the environment
# On macOS/Linux:
source attrition_env/bin/activate
# On Windows:
.\attrition_env\Scripts\activate
```

### 📥 Dependency Installation
All external frameworks—including `optuna`, `imblearn`, and `xgboost`—are pinned in the project registry. Install them simultaneously using the provided environment file:

```bash
pip install -r requirements.txt
```

### 💾 Direct Model Inference (Pre-trained Artifact)
If you wish to bypass the EDA and training pipeline entirely, the fully optimized and calibrated model is serialized and available for immediate downstream use:
*   **Artifact Format:** `model.pkl` (Python Pickle file)
*   **Usage:** You can directly load this binary object into your custom application or API endpoints using standard deserialization libraries to score production data instantly:
    ```python
    import joblib
    model = joblib.load('model.pkl')
    ```
---

## 📂 Repository Structure:
```text
├─ README.md
├─ requirements.txt
├─ app.py
├─ Project_report
├─ .gitignore.txt
├─ Data/
│    ├─ Palo Alto Networks.csv
│    ├─ Hyperparameters/
│    │        ├─ GB_params.json
│    │        ├─ LR_params.json
│    │        ├─ RF_params.json
│    │        └─ XGB_params.json
│    └─ Processed Employee Data/
│    │              ├─ X_train_final.csv
│    │              ├─ x_test_final.csv
│    │              ├─ Y_train_final.csv
│    │              └─ y_test_final.csv 
├─ Models/
│     ├─ classifier.pkl
│     └─ preprocessor_pipeline.pkl
├─ Notebooks/
│     ├─ EDA.ipynb
│     ├─ Feature_engineering.ipynb
│     ├─ model-optimizer.ipynb
│     └─model-training.ipynb
```





