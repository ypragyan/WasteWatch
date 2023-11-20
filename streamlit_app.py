import streamlit as st
import pandas as pd
import numpy as np

# Set page title and favicon
st.set_page_config(page_title="WasteWatch", page_icon="♻️")

# Function to load data
def load_data():
    # Replace this with your own data loading logic
    data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'Waste Type': np.random.choice(['Plastic', 'Paper', 'Glass'], 100),
        'Confidence': np.random.rand(100)
    })
    return data

# Function to create a data table
def display_data_table(data):
    st.subheader("Waste Sorting Data Table")
    st.dataframe(data)

# Function to create a sidebar with project information
def project_sidebar():
    st.sidebar.title("WasteWatch: AI Waste Sorting")
    st.sidebar.info(
        "WasteWatch is an AI-powered waste sorting project. It uses computer vision to classify waste types."
    )
    st.sidebar.subheader("Project")
    st.sidebar.markdown("[GitHub Repository](https://github.com/ypragyan/WasteWatch/tree/main)")
    st.sidebar.markdown("[Presentation](https://docs.google.com/presentation/d/1UNmV66jZEMEsP_Dm142TjZXEKSmYRRb8nxpoEgWEMis/edit#slide=id.g29cec67359b_0_0)")
    st.sidebar.subheader("Contact Information")
    st.sidebar.text("For inquiries, please contact:")
    st.sidebar.text("Pragyan Yadav")
    st.sidebar.text("ypragyan01@gmail.com")

# Main function
def main():
    st.title("WasteWatch: AI Waste Sorting")

    # Load data
    data = load_data()

    # Create layout
    project_sidebar()

    # Display project components
    display_data_table(data)

if __name__ == "__main__":
    main()
