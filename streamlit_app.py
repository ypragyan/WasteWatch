import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Set page title and favicon
st.set_page_config(page_title="WasteWatch: AI Waste Sorting", page_icon="♻️")

# Function to load data
def load_data():
    # Replace this with your own data loading logic
    data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'Waste Type': np.random.choice(['Plastic', 'Paper', 'Glass'], 100),
        'Confidence': np.random.rand(100)
    })
    return data

# Function to create a sidebar with project information
def project_sidebar():
    st.sidebar.title("WasteWatch: AI Waste Sorting")
    st.sidebar.info(
        "WasteWatch is an AI-powered waste sorting project. It uses computer vision to classify waste types."
    )
    st.sidebar.subheader("Project Links")
    st.sidebar.markdown("[GitHub Repository](https://github.com/yourusername/wastewatch)")
    st.sidebar.markdown("[Documentation](https://wastewatchdocs.com)")
    st.sidebar.subheader("Contact Information")
    st.sidebar.text("For inquiries, please contact:")
    st.sidebar.text("Your Name")
    st.sidebar.text("youremail@example.com")

# Function to create the Problem page
def problem_page():
    st.title("Waste Sorting Problem")

    st.write(
        "The improper disposal of waste is a significant environmental concern. "
        "Waste sorting is crucial to minimize the impact on the environment and enable recycling."
    )

    # Add more information about the waste sorting problem

# Function to create the AI page
def ai_page():
    st.title("WasteWatch AI")

    # AI Model Architecture Image
    st.subheader("Model Architecture")
    model_architecture_image = Image.open("model_architecture.png")
    st.image(model_architecture_image, caption="WasteWatch Model Architecture", use_column_width=True)

    # Data Examples
    st.subheader("Data Examples")
    data_example_images = [
        Image.open("data_example_1.jpg"),
        Image.open("data_example_2.jpg"),
        Image.open("data_example_3.jpg")
    ]
    st.image(data_example_images, caption=["Example 1", "Example 2", "Example 3"], use_column_width=True)

    # Heatmaps
    st.subheader("Heatmaps")
    heatmap_images = [
        Image.open("heatmap_1.jpg"),
        Image.open("heatmap_2.jpg"),
        Image.open("heatmap_3.jpg"),
        Image.open("heatmap_4.jpg")
    ]
    st.image(heatmap_images, caption=["Heatmap 1", "Heatmap 2", "Heatmap 3", "Heatmap 4"], use_column_width=True)

# Main function
def main():
    # Load data
    load_data()

    # Create layout
    project_sidebar()

    # Create navigation
    pages = {"Problem": problem_page, "AI": ai_page}
    selection = st.sidebar.radio("Select Page", list(pages.keys()))

    # Display the selected page
    pages[selection]()

if __name__ == "__main__":
    main()
 
