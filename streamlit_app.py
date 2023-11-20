import streamlit as st
from PIL import Image

# Set page title and favicon
st.set_page_config(page_title="WasteWatch", page_icon="♻️")

# Function to create a sidebar with project information
def project_sidebar():
    st.sidebar.title("WasteWatch")
    st.sidebar.info("WasteWatch is an AI-powered waste sorting project")
    st.sidebar.subheader("Project Links")
    st.sidebar.markdown("[GitHub Repository](https://github.com/ypragyan/WasteWatch/tree/main)")
    st.sidebar.markdown("[Presentation](https://docs.google.com/presentation/d/1UNmV66jZEMEsP_Dm142TjZXEKSmYRRb8nxpoEgWEMis/edit#slide=id.g29cec67359b_0_0)")

# Function to create the AI page
def ai_page():
    st.title("WasteWatch AI")

    # AI Model Architecture Image
    st.subheader("Model Architecture")
    model_architecture_image = Image.open("Images/Model.png")
    st.image(model_architecture_image, caption="WasteWatch Model Architecture", use_column_width=True)
    st.write(" The model used in this project is designed in a sequential architecture for the task of image classification. It consists of several types of layers. Initially, Conv2D layers with 128 filters process the input image, followed by batch normalization for model stability. Max-pooling reduces spatial dimensions, and dropout helps prevent overfitting. The architecture repeats with 128 filters. The GlobalAveragePooling2D layer averages spatial dimensions, and Dense layers shape the final output. In this model, there are 6 output units, for classifying the input image into 6 types of waste.")
    # Data Examples
    st.subheader("Results")
    data_example_images = [
        Image.open("Images/Accuracy.png"),
        Image.open("Images/Loss.png"),
        Image.open("Images/wastewatch.jpg"),
    ]
    st.image(data_example_images, caption=["Example 1", "Example 2", "Example 3"], use_column_width=True)

    # Heatmaps
    st.subheader("Attribution Methods: Visualizing AI gained Knowledge")
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
    pages = {"AI": ai_page}
    selection = st.sidebar.radio("Page", list(pages.keys()))

    # Display the selected page
    pages[selection]()

if __name__ == "__main__":
    main()
