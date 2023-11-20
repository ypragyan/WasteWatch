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
# Function to create the Problem page
def problem_page():
    st.title("Waste Sorting Problem")
    st.text("While the waste sorting industry is an essential part of our environment, we fail to pay adequate attention to the problems it causes in the lives of people that are directly involved in the process such as laborers who manually sort waste. Their lives are often spent in poverty and bio-hazardous conditions because of their occupation.")
    st.text("Manual waste pickers in the industry are no longer equipped to handle the 2 trillion tonnes of waste produced each year globally. "
            "Waste pickers live on the street or on dump sites. In addition, waste pickers are mobile and their work often varies seasonally.")
    
    st.text("Not efficiently segregating waste can cause economic issues, like wasting land resources, and environmental problems, "
            "such as the release of harmful gases from piled-up waste.")

    st.text("Waste pickers’ earnings vary widely between regions depending on different factors. Incomes are as low as US $2/day in some areas.")

    st.text("The majority of waste pickers have generally low levels of formal education. In many places, the work is done by primarily disadvantaged groups.")

    st.text("Informal waste management sector disproportionately focuses on lower-tier women, while men dominate higher-income roles, "
            "resulting in a gendered division of labor and exclusion from formalized activities.")


# Function to create the AI page
def ai_page():
    st.title("WasteWatch AI")

    # AI Model Architecture Image
    st.subheader("Model Architecture")
    model_architecture_image =  Image.open("Images/Model.png")
    st.image(model_architecture_image, caption="WasteWatch Model Architecture", use_column_width=True)

    # Data Examples
    st.subheader("Results")
    data_example_images = [
        Image.open("Images/Accuracy.png"),
        Image.open("Images/Loss.png"),
        Image.open("Images/wastewatch.jpg"),
    ]
    st.image(data_example_images, caption=["Example 1", "Example 2", "Example 3"], use_column_width=True)

    # Heatmaps
    st.subheader("Attribution Methods: Visualising AI gained Knowledge")
    heatmap_images = [
    Image.open("Images/Heatmap.png"),
    Image.open("Images/Layers.png"),
    Image.open("Images/XAI.png"),
 
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
 
