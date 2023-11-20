# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
pip install streamlit pandas numpy Pillow tensorflow
# Creating a simple Streamlit app
def main():
    st.title("Simple Streamlit App")

    # Sidebar with user input
    user_input = st.sidebar.text_input("Enter your name", "John Doe")
    st.sidebar.write(f"Hello, {user_input}!")

    # Main content
    st.header("Data Exploration")

    # Create a sample DataFrame
    data = pd.DataFrame({
        'Numbers': np.random.randn(100),
        'Categories': np.random.choice(['A', 'B', 'C'], 100)
    })

    # Display the DataFrame
    st.write("Sample DataFrame:")
    st.write(data)

    # Plot a histogram of the 'Numbers' column
    st.bar_chart(data['Numbers'])

    # Display a map with random data points
    map_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.7749, -122.4194],
        columns=['lat', 'lon']
    )
    st.map(map_data)

if __name__ == "__main__":
    main()
