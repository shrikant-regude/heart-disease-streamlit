import streamlit as st 
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')  

st.title("Heart Disease Prediction by Shrikant💗")
st.markdown("### Please enter the following details:")

age= st.slider('Age', 18, 100, 30)
sex = st.selectbox('Sex', ['Male', 'Female'])
chest_pain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp= st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
chol = st.slider('Cholesterol', 100, 600, 200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
max_heart_rate = st.slider('Maximum Heart Rate Achieved', 60, 200, 150)
exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])  
oldpeak = st.slider('Oldpeak', 0.0, 6.0, 1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping']) 


if st.button('Predict'):    
    input_data = {
        'age': age,
        'sex': sex,
        'chest_pain': chest_pain,
        'resting_bp': resting_bp,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'max_heart_rate': max_heart_rate,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope
    }

    input_df = pd.DataFrame([input_data])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[expected_columns]
    input_scaled = scaler.transform(input_df) 
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("The model predicts that you have heart disease. Please consult a healthcare professional for further evaluation.")
    else:
        st.success("The model predicts that you do not have heart disease. However, please remember that this is just a prediction and not a diagnosis. Always consult a healthcare professional for any health concerns.")