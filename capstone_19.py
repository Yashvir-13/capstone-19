import streamlit as st
import numpy as np
import pandas as pd
import diabetes_home,diabetes_predict,diabetes_plots
st.set_page_config(page_title = 'Early Diabetes Prediction Web App',page_icon = 'random',layout = 'wide',initial_sidebar_state = 'auto')
@st.cache(suppress_st_warning=True)
def load_data():
    df = pd.read_csv('diabetes.csv')
    df.head()
    df.rename(columns = {"BloodPressure": "Blood_Pressure",}, inplace = True)
    df.rename(columns = {"SkinThickness": "Skin_Thickness",}, inplace = True)
    df.rename(columns = {"DiabetesPedigreeFunction": "Pedigree_Function",}, inplace = True)
    df.head() 
    return df
df = load_data()
pages_dict = {"Home": diabetes_home,"Predict Diabetes": diabetes_predict,"Visualise Decision Tree": diabetes_plots}
st.sidebar.title("Navigation")
user_selection=st.sidebar.radio("Go to",tuple(pages_dict.keys()))
if user_selection=="Home":
    diabetes_home.app(df)
else:
    pages_dict[user_selection].app(df)  

