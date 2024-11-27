import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"./diabetes.csv")

# Enhanced Header Design
st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #FF9F40;
            font-family: 'Arial', sans-serif;
            padding: 10px 0;
            border-bottom: 2px solid #FF9F40;
        }
        .sidebar-form label {
            font-weight: bold;
            color: #333;
        }
        .prediction-output {
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            margin-top: 10px;
        }
    </style>
    <h1 class="main-header">Diabetes Prediction App</h1>
    <p style="text-align: center;">This app predicts whether a patient is diabetic based on their health data.</p>
""", unsafe_allow_html=True)

st.markdown("---")
st.sidebar.header('Enter Patient Data')

def calc():
    with st.sidebar.form("user_data_form"):
        pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17, value=3)
        bp = st.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
        bmi = st.number_input('BMI', min_value=0, max_value=67, value=20)
        glucose = st.number_input('Glucose', min_value=0, max_value=200, value=120)
        skinthickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
        dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47)
        insulin = st.number_input('Insulin', min_value=0, max_value=846, value=79)
        age = st.number_input('Age', min_value=21, max_value=88, value=33)
        submitted = st.form_submit_button("Submit")
        
    output = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bp': bp,
        'skinthickness': skinthickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age
    }
    return pd.DataFrame(output, index=[0]) if submitted else None

user_data = calc()
if user_data is not None:
    st.subheader('Patient Data Summary')
    st.write(user_data)

    x = df.drop(['Outcome'], axis=1)
    y = df['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    progress = st.progress(0)
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    progress.progress(100)

    result = rf.predict(user_data)

    output = 'You are not Diabetic' if result[0] == 0 else 'You are Diabetic'
    color = '#4CAF50' if result[0] == 0 else '#FF4136'
    
    st.markdown(f"<div class='prediction-output' style='background-color: {color}; color: white;'>{output}</div>", unsafe_allow_html=True)

    accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
    st.subheader('Model Accuracy:')
    st.write(f"{accuracy:.2f}%")
