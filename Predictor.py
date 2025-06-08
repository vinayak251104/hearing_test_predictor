import numpy as np 
import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt 
import joblib

st.markdown(
    """
    <style>
        /* Background */
        .stApp {
            background-color: #FFFFFF;  /* Pure White */
            color: #000000;  /* Black Text */
        }

        /* Top Navigation Bar */
        header {
            background-color: #E0E0E0 !important; /* Light Grey Tint */
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #F8F8F8 !important; /* Very Light Grey */
            color: #000000 !important; /* Black Text */
        }

        /* Sidebar Text */
        section[data-testid="stSidebar"] * {
            color: #000000 !important; /* Black Text */
            font-weight: bold !important;
        }

        /* Titles & Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #000000 !important; /* Black */
            font-weight: bold;
        }

        /* Data Editor (Table) */
        div[data-testid="stDataFrame"] {
            background-color: #F8F8F8 !important; /* Light Grey */
            color: #000000 !important; /* Black Text */
            border-radius: 10px;
        }

        /* Sliders */
        div[data-baseweb="slider"] {
            color: #000000 !important;  /* Black Text */
        }

        /* Text Above Sliders */
        div[data-testid="stSliderLabel"] {
            color: #000000 !important; /* Black */
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)
model=joblib.load('hearing_test_model.pkl')
scaler=joblib.load('scaler_transform_hearing.pkl')
st.title('HEARING TEST PREDICTOR')

col1,col2=st.columns(2)
with col1:
    st.subheader('Person Information')
    age=st.slider('Enter your Age',18,90)
    physical_score=st.slider('Enter your Physical Score',0,50)
    button=st.button('Predict')

with col2:
    st.subheader('Prediction Result:')
    new_data=scaler.transform([[age,physical_score]])
    pred=model.predict(new_data)
    predicted_text = pred[0]  
    if button:
        if predicted_text=='Pass':
            st.write('**Based on Past Trends, you will most likely pass the test!**')
        elif predicted_text=='Fail':
            st.write('**Based on Past Trends, you will most likely fail the test!**')


st.subheader("Hearing Test Recommendation")

st.write("**Pass:**")
st.write("- Maintain good ear hygiene.")
st.write("- Protect ears from excessive loud noise.")
st.write("- Regular check-ups if exposed to noisy environments.")

st.write("**Fail:**")
st.write("- Consult an audiologist for further evaluation.")
st.write("- Avoid loud environments and use hearing protection.")
st.write("- Consider a follow-up test for confirmation.")








       

