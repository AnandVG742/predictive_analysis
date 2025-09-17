import pandas as pd
import streamlit as st
import os
import plotly.express as px
from joblib import load
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Hospital Readmission Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This is a Streamlit app to analyze and predict hospital readmissions."
    }
)

# --- Path Configuration ---
DATA_PATH = 'hospital_readmissions.csv'
MODEL_PATH = 'readmission_model.joblib'
MODEL_METADATA_PATH = 'model_metadata.joblib'

# --- Load Data & Model (with caching) ---
@st.cache_data
def load_data(file_path):
    """Loads the data from the CSV file."""
    if not os.path.exists(file_path):
        st.error(f"Error: The file '{file_path}' was not found.")
        st.stop()
    return pd.read_csv(file_path)

@st.cache_resource
def load_model_and_metadata(model_path, metadata_path):
    """Loads the pre-trained model and its metadata."""
    if not os.path.exists(model_path):
        st.error("Error: The trained model file 'readmission_model.joblib' was not found.")
        st.info("Please run `python train_model.py` in your terminal to create the model file.")
        st.stop()
    if not os.path.exists(metadata_path):
        st.error("Error: The model metadata file 'model_metadata.joblib' was not found.")
        st.info("Please run `python train_model.py` in your terminal to create this file.")
        st.stop()

    model = load(model_path)
    metadata = load(metadata_path)
    return model, metadata

# Load everything at the start
try:
    df = load_data(DATA_PATH)
    model, metadata = load_model_and_metadata(MODEL_PATH, MODEL_METADATA_PATH)

    # --- Data Preprocessing for Dashboard ---
    df['age'] = df['age'].str.replace(')', '').str.replace('[', '').str.replace('-', ' to ')
    df['readmitted_binary'] = df['readmitted'].apply(lambda x: 'Readmitted' if x == 'yes' else 'Not Readmitted')

    # --- Sidebar Navigation ---
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Descriptive Analysis", "Patient Risk Analysis"])

    # --- Descriptive Analysis Page ---
    if page == "Descriptive Analysis":
        st.title("📈 Hospital Readmission Analysis Dashboard")
        st.markdown(
            """
            Explore key factors influencing patient readmission rates using an interactive dashboard.
            """
        )

        st.divider()

        # Sidebar Filters
        st.sidebar.header("Dashboard Filters")
        age_groups = sorted(df['age'].unique())
        selected_age_groups = st.sidebar.multiselect(
            "Filter by Age Group",
            options=age_groups,
            default=age_groups
        )

        specialties = sorted(df['medical_specialty'].unique())
        selected_specialties = st.sidebar.multiselect(
            "Filter by Medical Specialty",
            options=specialties,
            default=specialties
        )

        filtered_df = df[
            df['age'].isin(selected_age_groups) &
            df['medical_specialty'].isin(selected_specialties)
        ]

        # Top-level Metrics
        total_patients = len(filtered_df)
        readmitted_patients = filtered_df[filtered_df['readmitted_binary'] == 'Readmitted']
        readmission_rate = (len(readmitted_patients) / total_patients) * 100 if total_patients > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patients", f"{total_patients:,}")
        with col2:
            st.metric("Readmitted Patients", f"{len(readmitted_patients):,}")
        with col3:
            st.metric("Readmission Rate", f"{readmission_rate:.2f}%")

        st.divider()

        # Visualizations
        st.header("Readmission Trends")

        st.subheader("Readmission Rate by Age Group")
        age_readmission_rate = filtered_df.groupby('age')['readmitted_binary'].apply(lambda x: (x == 'Readmitted').mean() * 100).reset_index()
        fig1 = px.bar(
            age_readmission_rate,
            x='age',
            y='readmitted_binary',
            title='Readmission Rate by Age Group',
            labels={'readmitted_binary': 'Readmission Rate (%)', 'age': 'Age Group'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Top 10 Medical Specialties by Readmission Rate")
        specialty_readmission_rate = filtered_df.groupby('medical_specialty')['readmitted_binary'].apply(lambda x: (x == 'Readmitted').mean() * 100).reset_index()
        top_specialties = specialty_readmission_rate.sort_values(by='readmitted_binary', ascending=False).head(10)
        fig2 = px.bar(
            top_specialties,
            x='medical_specialty',
            y='readmitted_binary',
            title='Top 10 Specialties with Highest Readmission Rates',
            labels={'readmitted_binary': 'Readmission Rate (%)', 'medical_specialty': 'Medical Specialty'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Overall Readmission Distribution")
        readmission_counts = filtered_df['readmitted_binary'].value_counts()
        fig3 = px.pie(
            names=readmission_counts.index,
            values=readmission_counts.values,
            title='Proportion of Readmitted vs Not Readmitted Patients',
            color=readmission_counts.index,
            color_discrete_map={'Readmitted': '#ef553b', 'Not Readmitted': '#636efa'}
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.divider()
        with st.expander("View Raw Data"):
            st.dataframe(filtered_df, use_container_width=True)

    # --- Patient Risk Analysis Page ---
    else:
        st.title("👨‍⚕️ Patient Readmission Risk Analysis")
        st.markdown(
            """
            Input a patient's information below to predict their risk of being readmitted within 30 days.
            """
        )
        
        st.divider()

        st.sidebar.header("Input Patient Data")
        
        # Create input widgets for the features
        time_in_hospital = st.sidebar.slider("Time in Hospital (days)", 1, 14, 5)
        n_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 0, 150, 40)
        n_procedures = st.sidebar.slider("Number of Procedures", 0, 6, 2)
        n_medications = st.sidebar.slider("Number of Medications", 1, 81, 20)
        
        # Use the categorical features from the training metadata
        medical_specialty = st.sidebar.selectbox("Medical Specialty", options=metadata['medical_specialty_classes'])
        change = st.sidebar.selectbox("Change in Medications", options=metadata['change_classes'])
        diabetes_med = st.sidebar.selectbox("Diabetes Medication", options=metadata['diabetes_med_classes'])

        # Create a button to run the prediction
        if st.sidebar.button("Predict Readmission Risk", type="primary"):
            # Create a DataFrame from the input data
            input_data = pd.DataFrame([{
                'time_in_hospital': time_in_hospital,
                'n_lab_procedures': n_lab_procedures,
                'n_procedures': n_procedures,
                'n_medications': n_medications,
                'medical_specialty': medical_specialty,
                'change': change,
                'diabetes_med': diabetes_med
            }])
            
            # Make the prediction and get the probability
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            readmission_risk = prediction_proba[1] * 100

            # Display the result
            st.subheader("Prediction Result")
            colA, colB = st.columns([1, 2])
            
            with colA:
                st.metric(label="Predicted Readmission Risk", value=f"{readmission_risk:.2f}%")

            with colB:
                st.write("Based on the provided information, the patient's predicted outcome is:")
                if prediction == 1:
                    st.markdown("<p style='font-size: 24px; color: red; font-weight: bold;'>Readmitted within 30 days</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='font-size: 24px; color: green; font-weight: bold;'>Not Readmitted within 30 days</p>", unsafe_allow_html=True)

            st.info("Note: This is a predictive model based on historical data. It should not be used as a substitute for professional medical advice.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please ensure that you have run the `train_model.py` script to generate the model files.")
