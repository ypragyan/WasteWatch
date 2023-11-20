import streamlit as st
import pandas as pd
import numpy as np

# Set page title and favicon
st.set_page_config(page_title="Project Showcase", page_icon="🚀")

# Function to load data
def load_data():
    # Replace this with your own data loading logic
    data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'Value': np.random.randn(100).cumsum()
    })
    return data

# Function to create a data table
def display_data_table(data):
    st.subheader("Project Data Table")
    st.dataframe(data)

# Function to create a sidebar with project information
def project_sidebar():
    st.sidebar.title("Project Details")
    st.sidebar.info(
        "This is a showcase of a project. Provide a brief description of your project here."
    )
    st.sidebar.subheader("Project Links")
    st.sidebar.markdown("[GitHub Repository](https://github.com/yourusername/yourproject)")
    st.sidebar.markdown("[Documentation](https://yourprojectdocs.com)")
    st.sidebar.subheader("Contact Information")
    st.sidebar.text("For inquiries, please contact:")
    st.sidebar.text("Your Name")
    st.sidebar.text("youremail@example.com")

# Main function
def main():
    st.title("Project Showcase")

    # Load data
    data = load_data()

    # Create layout
    project_sidebar()

    # Display project components
    display_data_table(data)

if __name__ == "__main__":
    main()
