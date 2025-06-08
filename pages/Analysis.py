import numpy as np 
import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt 
import joblib
import plotly.express as px
import seaborn as sns
from sklearn.metrics import accuracy_score
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

st.title('Hearing Data/Model Analysis')
df=pd.read_csv('hearing_test_text.csv')
df3=pd.read_csv('hearing_test.csv')
df=df.drop('Unnamed: 0',axis=1)
df2=df3.corr(numeric_only=True)
df2=df2.fillna(0)
x=df.drop('test_result_text',axis=1)
y=df['test_result_text']

model=joblib.load('hearing_test_model.pkl')
scaler=joblib.load('scaler_transform_hearing.pkl')
x=scaler.transform(x)
pred=model.predict(x)
score=accuracy_score(y,pred)
accuracy=(score)*100

col1,col2=st.columns(2)

with col1:
    st.subheader('Correlation Heatmap')
    fig = px.imshow(
        df2,
        text_auto=True,  # Show correlation values
        aspect="auto",
        color_continuous_scale=px.colors.sequential.Viridis,  # Correct way!
        
    )

    # ✅ Adjust width & height for a wider display
    fig.update_layout(
        autosize=False,
        width=500,  # Wider matrix
        height=400  # Taller matrix
    )

    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("KDE Plot")
    selected_column = st.selectbox("Select a column for KDE plot:", df3.select_dtypes(include=['float64', 'int64']).columns)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(df3[selected_column], fill=True, color='blue', ax=ax)
    st.pyplot(fig)

col3,col4=st.columns(2)

with col3:
    st.subheader("Scatterplot")
    x_column = st.selectbox("Select X-axis:", df3.select_dtypes(include=['float64', 'int64']).columns)
    y_column = st.selectbox("Select Y-axis:", df3.select_dtypes(include=['float64', 'int64']).columns)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df[x_column], y=df[y_column], hue=df3["test_result"], palette="viridis", ax=ax)  
    st.pyplot(fig)
with col4:
    st.subheader("Barplot")
    category_column = st.selectbox("Select a Categorical Column:", df3.select_dtypes(include=['float64','int64']).columns)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=category_column, palette="viridis", ax=ax,data=df3,hue='test_result')
    plt.xticks(rotation=45)
    st.pyplot(fig)

col5,col6=st.columns(2)
with col5:
    st.subheader('Model Accuracy')
    st.write(f"the accuracy of our model is {accuracy} %")
    


