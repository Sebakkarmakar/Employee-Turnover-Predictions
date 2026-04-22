import streamlit as st
import joblib
model=joblib.load(r"C:\Users\admin\Desktop\ML PROJECTS\employee_turnover_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Employee Turnover", page_icon="👨‍💼")

# ---------- CSS Styling ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right,  #ffcccc, #add8e6);
}

h1 {
    text-align: center;
    color: #1565c0;
    font-size: 45px;
}

.sidebar .sidebar-content {
    background-color: #e3f2fd,!important
}

label {
    font-weight: bold;
    color: #0d47a1;
}

.stButton > button {
    background-color: #1565c0;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("""
<h1 style='color:red; font-size:30px;'>
👨‍💼 Employee Turnover Prediction App
</h1>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📋 Enter Employee Details")

    Job_Satisfaction = st.slider("Job Satisfaction", 1, 4)
    Performance_Rating = st.slider("Performance Rating", 1, 5)
    Years_At_Company = st.number_input("Years at Company", 0, 40)
    Work_Life_Balance = st.slider("Work Life Balance", 1, 4)
    Distance_From_Home = st.number_input("Distance From Home", 0, 50)
    Monthly_Income = st.number_input("Monthly Income", 1000, 100000)
    Education_Level = st.selectbox("Education Level", [1,2,3,4,5])
    Age = st.number_input("Age", 18, 60)
    Num_Companies_Worked = st.number_input("Companies Worked", 0, 10)

    Employee_Role = st.selectbox("Employee Role", ["Manager","Engineer","Sales","HR"])
    Department = st.selectbox("Department", ["HR","Sales","IT","Finance"])

    Annual_Bonus = st.number_input("Annual Bonus", 0, 50000)
    Training_Hours = st.number_input("Training Hours", 0, 100)

# ---------- Feature Engineering ----------
Annual_Bonus_Squared = Annual_Bonus ** 2
Interaction = Annual_Bonus * Training_Hours

# ---------- Predict Button ----------
import numpy as np
if st.button("🔍 Predict Turnover"):

    role_map = {"Manager":0, "Engineer":1, "Sales":2, "HR":3}
    dept_map = {"HR":0, "Sales":1, "IT":2, "Finance":3}

    Employee_Role_enc = role_map[Employee_Role]
    Department_enc = dept_map[Department]

    Annual_Bonus_Squared = Annual_Bonus ** 2
    Interaction = Annual_Bonus * Training_Hours

    input_data = np.array([[ 
        Job_Satisfaction,
        Performance_Rating,
        Years_At_Company,
        Work_Life_Balance,
        Distance_From_Home,
        Monthly_Income,
        Education_Level,
        Age,
        Num_Companies_Worked,
        Employee_Role_enc,
        Annual_Bonus,
        Training_Hours,
        Department_enc,
        Annual_Bonus_Squared,
        Interaction
    ]])

    # 🔥 THIS IS THE KEY LINE
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Employee likely to leave")
    else:
        st.success("✅ Employee likely to stay")