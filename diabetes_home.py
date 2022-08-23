import streamlit as st
@st.cache()
def app(df):
	st.title("Early Diabetes Prediction Web App")
	st.subheader("Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy.There isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help in reducing the impact of diabetes.This Web app will help you to predict whether a person has diabetes or is prone to get diabetes in future by analysing the values of several features using the Decision Tree Classifier.")
	with st.beta_expander("View Dataset"):
		st.dataframe(df)
	st.subheader("Columns's description")
	if st.checkbox("Show summary"):
		st.write(df.describe())
	col_1,col_2,col_3=st.beta_columns(3)
	with col_1:
		if st.checkbox("Show column names"):
			st.write(df.columns)
	with col_2:
	    if st.checkbox("Show datatypes"):
	        st.write(df.dtypes)
	with col_3:
	    if st.checkbox("Show column data"):
	        col=st.selectbox("Select the column",tuple(df.columns))
	        st.write(df[col]) 
