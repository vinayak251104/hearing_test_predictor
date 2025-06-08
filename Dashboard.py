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
st.title('HEARING TEST DASHBOARD')



st.header('HEARING TEST PREDICTOR')

st.subheader("🔹 About the App")
st.write(
    """
    Welcome to the **Hearing Test Predictor**, a simple yet effective tool designed to assess 
    hearing capability based on key factors. This app uses a **machine learning model** trained 
    on age and physical score to predict potential hearing concerns.
    """
)

st.subheader("🔹 How It Works")
st.write(
    """
    Our model, built using **Logistic Regression**, analyzes your input data and provides an 
    instant prediction. The app ensures accuracy by normalizing input values with **StandardScaler**, 
    optimizing the model through **GridSearchCV**, and selecting the best regularization method (**L1 penalty**).
    """
)

st.subheader("🔹 Features")
st.write(
    """
    ✔ **Predictor Page** – Input your details and get a real-time prediction.  
    ✔ **Analysis Page** – Gain insights into model accuracy and performance.  
    ✔ **Overview Page** – Learn about the dataset, methodology, and model selection.  
    """
)

st.write(
    "This app is a demonstration of **machine learning in healthcare**, providing a user-friendly "
    "interface to showcase predictive analytics. 🚀"
)

col1,col2=st.columns(2)
with col1:
    st.header('Sample Data Used')
    df=pd.read_csv('hearing_test.csv')
    st.write(df.head(100))
with col2:
    st.header('Data Description')
    st.write(
    """
    **Prediction Output:**  
    - **1:** Test Passed  
    - **0:** Test Failed  
    """
    )
    st.write(f"**Average Age:** {df['age'].mean()}")

    st.write(f"**Average Physical Score:** {df['physical_score'].mean()}")


    st.write(f"**Max Physical Score:** {df['physical_score'].max()}")

    st.write(f"**Max Age:** {df['age'].max()}")

    st.write(f"**Min Physical Score:** {df['physical_score'].min()}")

    st.write(f"**Min Age:** {df['age'].min()}")


    





 

                               
