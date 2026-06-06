import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import os
import lime
import lime.lime_tabular

st.set_page_config(page_title= "Attrition.app",layout="wide")

st.markdown("""
    <style>
        /* Removes huge empty padding space at the top of the main container */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
        }
        /* Removes empty whitespace block sitting right above the main title */
        stHeader {
            height: 0px !important;
        }
    </style>
""", unsafe_allow_html=True)

#------------------------------
# Data and model loading
#------------------------------
try:
    classifier = joblib.load("Models/classifier.pkl")
except ModuleNotFoundError as e:
    st.error(f"Missing Library Detected: {e}")
    st.stop()
except Exception as e:
    st.error("Model Error Occurred:")
    st.code(traceback.format_exc())

try:
    preprocessor = joblib.load("Models/preprocessor_pipeline.pkl")
except ModuleNotFoundError as e:
    st.error(f"Missing Library Detected: {e}")
except Exception as e:
    st.error("Preprocessor Error Occurred:")
    st.code(traceback.format_exc())

#---------------------------------------------------------------
# Functions
#--------------------------------------------------------------

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "dashboard"

def go_to_simulator():
    st.session_state["current_page"] = "simulator"

def go_to_dashboard():
    st.session_state["current_page"] = "dashboard"

if "employee_data" not in st.session_state:
    st.session_state["employee_data"] = None

def create_features(df):
    df = df.copy()
    satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
    
    # Note: Requires 'Attrition' in the dataset to calculate weights. 
    # For single-row inference, consider statically defining these weights in production.
    if 'Attrition' in df.columns:
        correlations = df[satisfaction_cols].corrwith(df['Attrition']).abs()
        weights = correlations / correlations.sum()
    else:
        # Fallback equal weights if target is missing (e.g., in inference)
        weights = pd.Series(0.25, index=satisfaction_cols)

    df['OverallSatisfaction'] = np.ceil(
        (weights['JobSatisfaction'] * df['JobSatisfaction']) +
        (weights['EnvironmentSatisfaction'] * df['EnvironmentSatisfaction']) +
        (weights['RelationshipSatisfaction'] * df['RelationshipSatisfaction']) +
        (weights['WorkLifeBalance'] * df['WorkLifeBalance'])
    ).astype(int)
    
    role_level_avg = df.groupby(['JobRole', 'JobLevel'])['MonthlyIncome'].transform('mean')
    # Fill NA just in case a single row grouping returns NaN
    role_level_avg = role_level_avg.fillna(df['MonthlyIncome']) 

    scaler = StandardScaler()
    cols = ['JobInvolvement', 'PerformanceRating', 'JobLevel']
    scaled_values = scaler.fit_transform(df[cols])
    df_scaled = pd.DataFrame(scaled_values, columns=cols, index=df.index)
    
    df['CompensationRatio'] = df['MonthlyIncome'] / role_level_avg
    df['IncomeExperienceRatio'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    df['PromotionDelay'] = df['YearsSinceLastPromotion'] / (df['YearsInCurrentRole'] + 1)
    df['IfStressed'] = np.where(
        (df['OverTime'] == 'Yes') & (df['BusinessTravel'] == 'Travel Frequently'), 1, 0
    )
    df['ManagerFriction'] = np.where(
        (df['YearsWithCurrManager'] < 1) & (df['YearsAtCompany'] > 3), 1, 0
    )
    df['JobChangeFrequency'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)
    df['LoyalityScore'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    df['RoleStagnationIndex'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1e-5)
    df['ExternalExperienceValue'] = (df['TotalWorkingYears'] - df['YearsAtCompany']) / df['JobLevel']
    
    # Handle single-row grouping NaNs with fillna
    edu_income_median = df.groupby('Education')['MonthlyIncome'].transform('median').fillna(df['MonthlyIncome'])
    df['EducationIncomeRatio'] = df['MonthlyIncome'] / edu_income_median
    
    df['CommuteBurnout'] = ((df['OverTime'] == 'Yes') & (df['DistanceFromHome'] > df['DistanceFromHome'].quantile(0.66))).astype('int')
    return df

@st.dialog("Upload Employee File")
def upload_file_dialog():
    st.write("Please select your employee HR data CSV file.")
    uploaded_file = st.file_uploader("Choose CSV", type=["csv"])
    
    if uploaded_file is not None:
        if st.button("Run"):
            st.session_state["employee_data"] = pd.read_csv(uploaded_file)
            go_to_dashboard()
            st.rerun()

#---------------------------------------------------------
# General Dashboard
#---------------------------------------------------------
if st.session_state["current_page"] == "dashboard":
    st.title("HR Analytics: Attrition Dashboard")
    btn_left_col, space_col, btn_right_col = st.columns([1.5, 6, 2.5], vertical_alignment="top")
    with btn_left_col:
        st.button("What-IF simulator", on_click=go_to_simulator)
    with btn_right_col:
        if st.button("📥 Import Employee CSV Data"):
            upload_file_dialog()
            
    if st.session_state["employee_data"] is not None:
        hr_data = st.session_state["employee_data"]
        
        # Adjust these columns based on your specific model requirements
        columns_need = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance', 
                        'JobRole', 'JobLevel', 'MonthlyIncome', 'JobInvolvement', 'PerformanceRating', 
                        'TotalWorkingYears', 'YearsSinceLastPromotion', 'YearsInCurrentRole', 'OverTime', 
                        'BusinessTravel', 'YearsWithCurrManager', 'YearsAtCompany', 'NumCompaniesWorked', 
                        'Education', 'DistanceFromHome', 'EmployeeId']
        
        columns_current_set = set(hr_data.columns)
        columns_need_set = set(columns_need)
        has_all_columns = columns_need_set.issubset(columns_current_set)
        
        if has_all_columns:
            df_process = hr_data.copy()
            df_process = create_features(df_process)
            
            # Preprocessing & Prediction
            processed_data = preprocessor.transform(df_process)
            probabilities = classifier.predict_proba(processed_data)
            attrition_risk_scores = probabilities[:, 1]
            
            results_df = pd.DataFrame({"EmployeeId": hr_data["EmployeeId"], "Attrition Risk Score": attrition_risk_scores})
            results_df["Attrition Risk Score"] = results_df["Attrition Risk Score"].map("{:.1%}".format)
            # ---KPI METRICS ---
            total_emp = len(results_df)
            numeric_risk = results_df["Attrition Risk Score"].str.rstrip('%').astype(float)
            high_risk = len(numeric_risk[numeric_risk > 70])
            avg_risk = numeric_risk.mean()

            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Total Employees", total_emp)
            kpi2.metric("High Flight Risk", high_risk)
            kpi3.metric("Average Risk Score", f"{avg_risk:.1f}%")
            st.markdown("---") 
            # -------------------------------
            
            col_left, col_right = st.columns([4, 6])
            with col_left:
                if "visible_rows" not in st.session_state:
                    st.session_state["visible_rows"] = 10
                current_limit = st.session_state["visible_rows"]
                sorted_df = results_df.sort_values(by="Attrition Risk Score", ascending=False)
                expanded_df = sorted_df.head(current_limit).copy()
                
                st.write(f"#### 🚨 Top {len(expanded_df)} Flight-Risk Employees")
                st.dataframe(expanded_df, use_container_width=True, height=150)
                
                total_available_rows = len(results_df)
                if current_limit < total_available_rows:
                    def load_more_employees():
                        st.session_state["visible_rows"] += 5
                    st.button("🔽 Click to see more", on_click=load_more_employees)
                else:
                    st.info("✨ Showing all available employee risk scores.")
                    
                st.markdown("<br>", unsafe_allow_html=True)
            
                st.write("#### 📊 Probability Distribution Visualization")
                fig_dist = px.histogram(
                    results_df, 
                    x="Attrition Risk Score", 
                    labels={"Attrition Risk Score": "Predicted Attrition Probability", "count": "Number of Employees"},
                    color_discrete_sequence=["#4A90E2"]
                )
                fig_dist.update_layout(yaxis_title="Count of Employees", height=250)
                st.plotly_chart(fig_dist, use_container_width=True)

            with col_right:
                importances = classifier.feature_importances_
                feature_names = preprocessor.get_feature_names_out()
                clean_feature_names = [name.split("__")[-1] for name in feature_names]
                
                df_importance = pd.DataFrame({
                    "Feature": clean_feature_names,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=True)
                
                st.write("#### Key Drivers of Employee Attrition")
                fig_importance = px.bar(
                    df_importance.tail(15),  # Display top 15 features
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    labels={"Importance": "Relative Importance Score", "Feature": "Employee Attribute"},
                    color="Importance",
                    color_continuous_scale="Reds"
                )
                fig_importance.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
                st.info(
                    "💡 **HR Insight:** This chart displays the global drivers of attrition risk. "
                    "Higher scores indicate that the feature has a stronger impact on an employee's decision to leave."
                )
            # --- SUNBURST CHART ---
            st.markdown("---")
            st.write("#### 🎯 Risk Concentration by Department & Role")
            merged_df = hr_data.merge(results_df, on="EmployeeId")
            merged_df["Risk Numeric"] = merged_df["Attrition Risk Score"].str.rstrip('%').astype(float)

            fig_sunburst = px.sunburst(
                merged_df, 
                path=['Department', 'JobRole'], 
                values='Risk Numeric',
                color='Risk Numeric', 
                color_continuous_scale='Reds'
            )
            fig_sunburst.update_layout(height=500)
            st.plotly_chart(fig_sunburst, use_container_width=True)
            # ----------------------------------
        else:
            missing_columns = columns_need_set - columns_current_set
            st.error(f"❌ Missing Columns! The uploaded file is missing: {list(missing_columns)}")
    else:
        st.write("Upload Employee data as .csv file. Ensure 'EmployeeId' feature is inside the CSV file.")


#----------------------------------------------------
# What-if Scenario simulator
#----------------------------------------------------
if st.session_state["current_page"] == "simulator":
    st.title("🧪 What-If Scenario Simulator")
    with st.sidebar:
        st.write("Adjust employee attributes to simulate attrition risk behavior.")
        st.button("General Dashboard", on_click=go_to_dashboard)
        
        # Collecting Inputs
        Age = st.number_input("Age", 18, 65, value=30)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        EducationField = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
        
        JobRole = st.selectbox("Job Role", options=["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
        JobLevel = st.slider("Job Level", 1, 5, value=2)
        MonthlyIncome = st.number_input("Monthly Income", value=5000)
        DailyRate = st.number_input("Daily Rate", value=800)
        HourlyRate = st.number_input("Hourly Rate", value=65)
        MonthlyRate = st.number_input("Monthly Rate", value=15000)
        PercentSalaryHike = st.number_input("Percent Salary Hike", 0, 50, value=15)
        
        TotalWorkingYears = st.number_input("Total Working Years", 0, 40, value=10)
        YearsAtCompany = st.number_input("Years at Company", 0, 40, value=5)
        YearsInCurrentRole = st.number_input("Years in Current Role", 0, 40, value=3)
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 40, value=1)
        YearsWithCurrManager = st.number_input("Years with Current Manager", 0, 40, value=2)
        
        JobSatisfaction = st.slider("Job Satisfaction", 1, 4, value=3)
        EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, value=3)
        RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, value=3)
        WorkLifeBalance = st.slider("Work Life Balance", 1, 4, value=3)
        
        JobInvolvement = st.slider("Job Involvement", 1, 4, value=3)
        PerformanceRating = st.slider("Performance Rating", 1, 4, value=3)
        
        OverTime = st.radio("Over Time", ["Yes", "No"])
        BusinessTravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        NumCompaniesWorked = st.number_input("Number of Companies Worked", 0, 10, value=2)
        Education = st.slider("Education Level", 1, 5, value=3)
        DistanceFromHome = st.number_input("Distance From Home", 1, 50, value=10)
        TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 6, value=2)
        StockOptionLevel = st.slider("Stock Option Level", 0, 3, value=0)

    input_data = {
        "Age": [Age],
        "Gender": [Gender],
        "MaritalStatus": [MaritalStatus],
        "Department": [Department],
        "EducationField": [EducationField],
        "JobRole": [JobRole],
        "JobLevel": [JobLevel],
        "MonthlyIncome": [MonthlyIncome],
        "DailyRate": [DailyRate],
        "HourlyRate": [HourlyRate],
        "MonthlyRate": [MonthlyRate],
        "PercentSalaryHike": [PercentSalaryHike],
        "TotalWorkingYears": [TotalWorkingYears],
        "YearsAtCompany": [YearsAtCompany],
        "YearsInCurrentRole": [YearsInCurrentRole],
        "YearsSinceLastPromotion": [YearsSinceLastPromotion],
        "YearsWithCurrManager": [YearsWithCurrManager],
        "JobSatisfaction": [JobSatisfaction],
        "EnvironmentSatisfaction": [EnvironmentSatisfaction],
        "RelationshipSatisfaction": [RelationshipSatisfaction],
        "WorkLifeBalance": [WorkLifeBalance],
        "JobInvolvement": [JobInvolvement],
        "PerformanceRating": [PerformanceRating],
        "OverTime": [OverTime],
        "BusinessTravel": [BusinessTravel],
        "NumCompaniesWorked": [NumCompaniesWorked],
        "Education": [Education],
        "DistanceFromHome": [DistanceFromHome],
        "TrainingTimesLastYear": [TrainingTimesLastYear],
        "StockOptionLevel": [StockOptionLevel]
    }
    
    df_sim = pd.DataFrame(input_data)
    
    # Feature Engineering
    df_sim = create_features(df_sim)
    
    # Prediction
    sim_preprocessed = preprocessor.transform(df_sim)
    
    if st.button("Predict Attrition Risk"):
        probability = classifier.predict_proba(sim_preprocessed)
        risk_percentage = probability[0][1] * 100
        
        if risk_percentage < 30:
            color = "#2ecc71"  # Soft Green
            status_label = "🟢 Low Flight Risk"
        elif risk_percentage < 70:
            color = "#f39c12"  # Soft Orange
            status_label = "🟡 Medium Flight Risk"
        else:
            color = "#e74c3c"  # Soft Red
            status_label = "🔴 High Flight Risk!"

        col1, col2 = st.columns([3, 7])
        with col1:
            st.write("### Prediction Results")
            st.markdown(f"""
                <div style="background-color: #1e222b; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
                    <p style="margin: 0; font-size: 14px; color: #a3a8b4; font-weight: bold; text-transform: uppercase;">{status_label}</p>
                    <h1 style="margin: 5px 0 0 0; font-size: 48px; color: {color}; font-weight: bold;">{risk_percentage:.2f}%</h1>
                </div>
            """, unsafe_allow_html=True)
            # --- INSERT GAUGE CHART HERE ---
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge",
                value=risk_percentage,
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(46, 204, 113, 0.2)'},
                        {'range': [30, 70], 'color': 'rgba(243, 156, 18, 0.2)'},
                        {'range': [70, 100], 'color': 'rgba(231, 76, 60, 0.2)'}
                    ],
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            # -------------------------------  

        with col2:
            st.write("### 🔍 LIME: Local Attrition Driver Breakdown")
            
            if st.session_state["employee_data"] is None:
                st.warning("⚠️ Please upload Employee Data in the General Dashboard to enable LIME explainability. LIME requires a background dataset to calculate local feature impacts.")
            else:
                # --- 1. PREPARE LIME BACKGROUND DATA ---
                hr_data_ref = st.session_state["employee_data"].copy()
                df_ref = create_features(hr_data_ref)
                X_ref = preprocessor.transform(df_ref)
                
                # Convert to dense array if your preprocessor outputs a sparse matrix
                if hasattr(X_ref, "toarray"):
                    X_ref = X_ref.toarray()
                    
                feature_names = [name.split("__")[-1] for name in preprocessor.get_feature_names_out()]
                
                # --- 2. INITIALIZE LIME EXPLAINER ---
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X_ref,
                    feature_names=feature_names,
                    class_names=['Stay', 'Attrition'],
                    mode='classification',
                    random_state=42
                )
                
                # Convert simulated data to dense array
                sim_dense = sim_preprocessed.toarray()[0] if hasattr(sim_preprocessed, "toarray") else sim_preprocessed[0]
                
                # --- 3. GENERATE EXPLANATION ---
                predict_fn = lambda x: classifier.predict_proba(x)
                exp = explainer.explain_instance(
                    data_row=sim_dense, 
                    predict_fn=predict_fn, 
                    num_features=8, # Limit to top 8 for UI cleanliness
                    labels=(1,)     # Force explanation for Class 1 (Attrition)
                )
                
                # --- 4. RENDER LIME WATERFALL CHART ---
                lime_list = exp.as_list(label=1)
                intercept = exp.intercept[1] * 100 
                
                dynamic_x = ["Background Average"]
                dynamic_y = [intercept]
                dynamic_text = [f"{intercept:.1f}%"]
                dynamic_measure = ["relative"]
                
                for condition, weight in lime_list:
                    feature_shift = weight * 100
                    dynamic_x.append(condition)  # LIME provides readable conditions like 'Age <= 30'
                    dynamic_y.append(feature_shift)
                    
                    prefix = "+" if feature_shift > 0 else ""
                    dynamic_text.append(f"{prefix}{feature_shift:.2f}%")
                    dynamic_measure.append("relative")
                    
                lime_pred = exp.local_pred[0] * 100
                dynamic_x.append("Final LIME Risk")
                dynamic_y.append(lime_pred)
                dynamic_text.append(f"{lime_pred:.2f}%")
                dynamic_measure.append("total")
                
                fig_waterfall = go.Figure(go.Waterfall(
                    name="LIME Attrition Drivers",
                    orientation="v",
                    measure=dynamic_measure,
                    x=dynamic_x,
                    text=dynamic_text,
                    y=dynamic_y,
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#e74c3c"}},  
                    decreasing={"marker": {"color": "#2ecc71"}},  
                    totals={"marker": {"color": "#4A90E2"}}       
                ))
                
                fig_waterfall.update_layout(
                    showlegend=False,
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_title="Attrition Risk Score (%)",
                    margin=dict(t=20, b=20, l=20, r=20),
                    xaxis={"tickangle": 45, "tickfont": {"size": 11}}
                )
                
                st.plotly_chart(fig_waterfall, use_container_width=True)
        # --- RADAR CHART ---
        st.write("### 🧭 Cultural & Satisfaction Alignment")
        categories = ['Job Satisfaction', 'Environment', 'Work-Life Balance', 'Job Involvement']
        emp_values = [JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, JobInvolvement]
        baseline_values = [2.7, 2.7, 2.7, 2.7]  # You can adjust these baselines

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=emp_values, theta=categories, fill='toself', name='Simulated Employee'))
        fig_radar.add_trace(go.Scatterpolar(r=baseline_values, theta=categories, fill='toself', name='Company Average'))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[1, 4])),
            showlegend=True,
            height=350,
            margin=dict(t=30, b=30, l=30, r=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        # -------------------------------
