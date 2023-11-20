import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Set page title and favicon
st.set_page_config(page_title="WasteWatch", page_icon="♻️")
 

# Function to create a sidebar with project information
def project_sidebar():
    st.sidebar.title("WasteWatch")
    st.sidebar.info(
        "WasteWatch is an AI-powered waste sorting project"
    )
    st.sidebar.subheader("Project Links")
    st.sidebar.markdown("[GitHub Repository](https://github.com/ypragyan/WasteWatch/tree/main)")
    st.sidebar.markdown("[Presentation](https://docs.google.com/presentation/d/1UNmV66jZEMEsP_Dm142TjZXEKSmYRRb8nxpoEgWEMis/edit#slide=id.g29cec67359b_0_0)")

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
    model_architecture_image =  Image.open("Images/Model.jpg")
    st.image(model_architecture_image, caption="WasteWatch Model Architecture", use_column_width=True)

    # Data Examples
    st.subheader("Results")
    data_example_images = [
        Image.open("Images/Accuray.jpg"),
        Image.open("Images/Loss.jpg"),
        Image.open("Images/wastewatch.jpg"),
    ]
    st.image(data_example_images, caption=["Example 1", "Example 2", "Example 3"], use_column_width=True)

    # Heatmaps
    st.subheader("Attribution Methods: Visualising AI gained Knowledge")
    heatmap_images = [
    Image.open("Images/Heatmap.jpg"),
    Image.open("Images/Layers.jpg"),
    Image.open("Images/XAI.jpg"),
 
    ]
    st.image(heatmap_images, caption=["Heatmap 1", "Heatmap 2", "Heatmap 3"], use_column_width=True)

# Main function
def main():
    # Load data 
    # Create layout
    project_sidebar()

    # Create navigation
    pages = {"Problem": problem_page, "AI": ai_page}
    selection = st.sidebar.radio("Select Page", list(pages.keys()))

    # Display the selected page
    pages[selection]()

if __name__ == "__main__":
    main()
 
