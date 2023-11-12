import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Waste Watch - AI for Waste Classification",
    page_icon="🗑️",
    layout="wide"
)


# Define custom CSS for the pastel blue background
custom_css = """
<style>
body {
    background-color: #b2d8d8; /* Pastel Blue */
}
</style>
"""

# Use st.markdown to apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Your Streamlit app content goes here
st.title("Pastel Blue Background")

# Add more content here


def overview_page():
    st.title("Project Overview")
    st.write("This is the overview page for your project.")
    # Add your project overview content here

# Create a function for another page
def other_page():
    st.title("Another Page")
    st.write("This is another page of your Streamlit app.")
    # Add content for this page here

# Create a navigation menu
page = st.sidebar.selectbox(
    "Select a page",
    ("Overview", "Interactive Demo", "Waste Management", "About Us")
)

# Define the function for the "Overview" page with custom styling
def overview_page():
    st.header("Project Overview")
    st.write("Waste Watch is an innovative AI model that helps automate the process of waste classification and sorting. It is designed to assist waste management and recycling facilities in identifying and segregating various types of waste materials efficiently. Our goal is to contribute to environmental sustainability and make waste management more efficient.")
    # Add more project overview details here

# Define the function for the "Interactive Demo" page with custom styling
def interactive_demo():
    st.header("Interactive Demo")
    st.write("Try out our interactive demo to see how the AI model can classify waste materials. Upload an image, and the AI will provide you with the classification results.")
    # Add the interactive demo code here

# Define the function for the "Waste Management" page with custom styling
def waste_management():
    st.header("Waste Management")
    st.write("Efficient waste management is crucial for a sustainable future. Waste Watch helps waste management facilities automate the sorting process, reducing errors and improving recycling rates.")
    # Add more information about waste management here

# Define the function for the "About Us" page with custom styling
def about_us():
    st.header("About Us")

    # Team member 1
    st.subheader("John Doe")
    st.image("https://intentplanning.ca/wp-content/uploads/2019/01/sample-person.jpg", caption="John Doe - Project Lead", use_column_width=True)
    st.text("John is passionate about AI and environmental sustainability. As the project lead, he oversees the development and implementation of Waste Watch.")

    # Team member 2
    st.subheader("Jane Smith")
    st.image("https://intentplanning.ca/wp-content/uploads/2019/01/sample-person.jpg", caption="Jane Smith - Machine Learning Engineer", use_column_width=True)
    st.text("Jane specializes in machine learning and is responsible for training and fine-tuning the AI model for waste classification.")

    # Team member 3
    st.subheader("Bob Johnson")
    st.image("https://intentplanning.ca/wp-content/uploads/2019/01/sample-person.jpg", caption="Bob Johnson - UX/UI Designer", use_column_width=True)
    st.text("Bob contributes to the project with his expertise in creating an intuitive and user-friendly interface for the Waste Watch application.")

    # Add more team members as needed

    # Contact information
    st.subheader("Contact Us")
    st.text("For inquiries or collaboration opportunities, please reach out to us at info@wastewatch.com.")

# Display the selected page
if page == "Overview":
    overview_page()
elif page == "Interactive Demo":
    interactive_demo()
elif page == "Waste Management":
    waste_management()
elif page == "About Us":
    about_us()




# Display the selected page
if page == "Overview":
    overview_page()
elif page == "Interactive Demo":
    interactive_demo()
elif page == "Waste Management":
    waste_management()
elif page == "About Us":
    about_us()
