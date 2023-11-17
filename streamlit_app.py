import streamlit as st
import tensorflow as tf


def overview_page():
    st.header("Project Overview")
    st.write("Waste Watch is an innovative AI model that helps automate the process of waste classification and sorting. It is designed to assist waste management and recycling facilities in identifying and segregating various types of waste materials efficiently. Our goal is to contribute to environmental sustainability and make waste management more efficient.")
    # Add more project overview details here
# Define the function for the "Interactive Demo" page with custom styling
def interactive_demo():
    # Add the interactive demo code here
    upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
    c1, c2= st.columns(2)
    if upload is not None:
      im= Image.open(upload)
      img= np.asarray(im)
      image= cv2.resize(img,(128, 128))
      img= preprocess_input(image)
      img= np.expand_dims(img, 0)
      c1.header('Input Image')
      c1.image(im)
      c1.write(img.shape)
      #load weights of the trained model.
 
# Define the function for the "Waste Management" page with custom styling
 
# Set page configuration
st.set_page_config(
    page_title="WasteWatch",
    page_icon="🗑️",
    layout="wide"
)
 
# Use st.markdown to apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)
# Front page layout
st.title("Welcome to Waste Watch!")
st.write("Empowering Waste Management with AI")
# Display an engaging image or illustration
st.image("Images/wastewatch.jpg", width= 100, caption="Image Source: Nidhi")
# Add a call-to-action button
 
page = st.sidebar.selectbox(
    "Select a page",
    ("Overview", "Interactive Demo")
)
# Display the selected page
if page == "Overview":
    overview_page()
elif page == "Interactive Demo":
    interactive_demo()
 
