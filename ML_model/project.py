import streamlit as st
import pandas as pd
import os
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

from pycaret.classification import pull,setup,compare_models,save_model

ml_img = "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.analyticsinsight.net%2Flatest-news%2Feverything-you-need-to-know-about-adversarial-machine-learning&psig=AOvVaw1vGIBgWyY-CDzN5wtESWo2&ust=1725285866486000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCLjU3Yj1oYgDFQAAAAAdAAAAABAE"

if os.path.exists("data.csv"):
    data = pd.read_csv("data.csv",index_col=None)

with st.sidebar:
    st.image(ml_img)
    st.title("Automatic ML")
    choice = st.radio("Features",["Upload","Profile","ML","Download"])
    st.info("This application will help you to build automated machine learning pipeline.")

if choice == "Upload":
    st.title("Upload Dataset")
    file = st.file_uploader("Upload your file for machine learning model.")
    if file:
        data = pd.read_csv(file,index_col=None)
        st.dataframe(data)
        data.to_csv("data.csv")

if choice == "Profile":
    if 'data' in locals():
        st.write("Data Analytics")
        report = ProfileReport(data)
        st_profile_report(report)

if choice == "ML":
    st.title("Machine Learning")
    target = st.selectbox("Select Your Target/Label",data.columns)

    setup(data,target=target)
    setup_df = pull()
    st.info("This is ML Experiment settings")
    st.dataframe(setup_df)

    best_model = compare_models()
    compare_df = pull()
    st.info("This is ML model")
    st.dataframe(compare_df)
    best_model
    save_model(best_model,"bestmodel")

if choice == "Download":
    st.title("Download")
    st.write("Best model for the given Dataset")
    with open("bestmodel.pkl","rb") as f:
        st.download_button("Download Model",f,"trainedmodel.pkl")
