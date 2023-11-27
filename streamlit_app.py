import streamlit as st
import tensorflow as tf
import requests
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image

# Title and page configuration
st.set_page_config(page_title="WasteWatch", page_icon="♻️", layout="wide")

# Load model, set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("model/")
    return model

with st.spinner("Loading Model...."):
    model = load_model()

# Classes for Waste Classification
classes = ["E-Waste", "Batteries", "Glass", "Metal", "Organic", "Paper"]

# Image preprocessing
def load_image(image):
    img = tf.image.decode_jpeg(image, channels=3)
    img = tf.cast(img, tf.float32)
    img /= 255.0
    img = tf.image.resize(img, (128, 128))
    img = tf.expand_dims(img, axis=0)
    return img

# Function to make Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

st.title("WasteWatch: AI-Powered Waste Sort")


page = st.sidebar.selectbox("Go to Page:", ["Explanation", "Demo"], index=1)
 

if page == "Explanation":
    
    st.subheader("Data")
    data_imgs = [
        Image.open("images/class.png"),
        Image.open("images/data.png"),
        Image.open("images/images.png")
    ]
    st.write("The dataset was obtained from Jay Soni, and it was categorized into six classes: e-waste, batteries, glass, metal, organic, and paper, representing the most common types of waste. While the class distribution appears relatively balanced overall, there is a noticeable disproportion in the amount of data allocated to paper and e-waste. This distribution closely mirrors real-world scenarios, where certain waste categories, such as paper, tend to be more prevalent, while others, like e-waste, are relatively less abundant.")
    st.write("In total, the dataset comprises 4,580 training samples and 1,128 testing samples. The images in the dataset exhibit a mean pixel value of 0.59, accompanied by a standard deviation of 0.292. These statistical measures provide insights into the overall brightness and variability of the images within the dataset.")
    st.image(data_imgs, caption=["Class Distribution Comparison ", "Sample Images of the dataset", "Pixel Value Distribution"])
    # AI Model Architecture Image
    st.subheader("Model")
    st.write(" The model used in this project is designed in a sequential architecture for the task of image classification. It consists of several types of layers. Initially, Conv2D layers with 128 filters process the input image, followed by batch normalization for model stability. Max-pooling reduces spatial dimensions, and dropout helps prevent overfitting. The architecture repeats with 128 filters. The Global Average Pooling 2D layer averages spatial dimensions, and Dense layers shape the final output. In this model, there are 6 output units, for classifying the input image into 6 types of waste.")
    st.write("The model is implemented using the TensorFlow and Keras frameworks. It utilizes the Adam optimizer with a learning rate of 1e-3 and is compiled with the Categorical Cross Entropy loss function. To enhance training, a ReduceLROnPlateau callback is employed, which dynamically adjusts the learning rate based on the validation loss, allowing for more efficient convergence. The training process involves 50 epochs, and the model's performance metrics, such as accuracy, are monitored. The training history, including information about the loss and accuracy during each epoch, is stored in the 'history' variable for further analysis.")
    model_architecture_image = Image.open("images/Model.png")
    st.image(model_architecture_image, caption="WasteWatch Model Architecture")
# Data Examples
    st.subheader("Results")
    data_example_images = [
        Image.open("images/Accuracy.png"),
        Image.open("images/Loss.png")
    ]
    st.image(data_example_images, caption=["Training and Validation Accuracy Over Epochs: The model achieves a peak accuracy of 78&.", "Training and Validation Loss Over Epochs"])
    
    st.write("We evaluate the performance of our models using several evaluation metrics: Recall, Accuracy, Precision and F1 score. True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). In the context of our paper, the FL class is the positive outcome and NF is the negative.")
    st.write("Recall gauges a model's capacity to identify all relevant instances of a positive class, crucial in scenarios with high costs for missing positive instances, such as medical diagnoses. Accuracy, representing overall correctness, can be misleading in imbalanced datasets, where it may not reflect performance on the minority class. Precision, assessing positive prediction accuracy, is vital when the cost of false positives is significant, as seen in spam email detection. F1 Score, the harmonic mean of precision and recall, offers a balanced evaluation, particularly beneficial when there's an uneven class distribution and consideration of both false positives and false negatives is essential.")
    st.write("To put the performance of our model in context, we establish a baseline for model evaluation by computing randomized labels. This baseline provides a point of reference to gauge how well our model performs, helping us assess the true predictive power of our model. As we can see in the table on the right, the WasteWatch model greatly outperforms the baseline model by achieve an accuracy of 78%. With an accuracy this high, our model is ready to be deployed in waste management facilities.")

    latex_code = r"""
    \begin{align}
    \text{Recall} & = \frac{TP}{TP + FN} \quad   \\
    \text{Accuracy} & = \frac{TP + TN}{TP + TN + FP + FN} \quad  \\
    \text{Precision} & = \frac{TP}{TP + FP} \quad   \\
    \text{F1-Score} & = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \quad  
    \end{align}
    """
    st.latex(latex_code)
    markdown_table_code = """
    | Model     | Accuracy | Precision | Recall | F1 Score |
    |-----------|----------|-----------|--------|----------|
    | Baseline  | 0.78     | 0.78      | 0.78   | 0.78     |
    | Our Model | 0.05     | 0.15      | 0.15   | 0.07     |
    """
    # Main content area
    st.markdown("Here is a comparison of model performance using various metrics.")
    st.markdown(markdown_table_code)
    st.write(" ")
    # Heatmaps
    st.subheader("Attribution Visualisation")
 
    
    # List of items for bullet points
    bullet_points = [
        "GradCAM (Gradient-weighted Class Activation Mapping): GradCAM is a model-agnostic method that leverages class-specific gradient information from the final convolutional layer of a Convolutional Neural Network (CNN) to generate a coarse localization map of crucial regions within the image.",
        "Guided Backpropagation operates on the premise that neurons act as detectors for specific image features. It calculates the gradient of the output concerning the input but, notably, only backpropagation non-negative gradients when passing through ReLU functions. This approach highlights the pixels that hold significance in the image.",
        "Guided Grad-CAM combines the fine-grained pixel importance details from guided backpropagation with the coarse localization advantages of GradCAM. It is computed as the element-wise product of guided backpropagation and the upsampled Grad-CAM attributions, providing a comprehensive method for highlighting relevant image regions while retaining fine-grained pixel importance information"
    ]
    
    # Display bullet points using HTML syntax
    st.markdown("<ul>" + "".join([f"<li>{item}</li>" for item in bullet_points]) + "</ul>", unsafe_allow_html=True)


    heatmap_images = [
        Image.open("images/Heatmap.png"),
        Image.open("images/Layers.png"),
        Image.open("images/XAI.png"),
    ]
    st.image(heatmap_images, caption=[
        "GradCAM Heatmap next to original image", 
        "GradCAM heatmap visualising the processing of an input image through the layers of the CNN model. This heat map illustrates the input image as it progresses through the various convolution layers of the model. This provides insights into how the model reaches decisions regarding waste classification, enhancing interpretability and transparency. ", 
        "The image illustrates various attribution methods, revealing that the model, when classying an image as e-waste, predominantly emphasizes rectangular edges."
    ])


# Demo page
if page == "Demo":
    
    # Sidebar with title and image source selection
    st.sidebar.title("Image Input Options")
    st.write("Upon uploading an image, our system will classify it into one of six categorie **e-waste, batteries, glass, metal, organic, and paper**. For detailed insights into the development of the AI model, explore the Explanations page.")    
    image_option = st.sidebar.radio("Select Image Source:", ("URL", "Upload"))
    st.sidebar.title("Instructions")
    if image_option == "URL":
        # Get image URL from user
        st.sidebar.write("Enter Image URL to classify:")
        image_path = st.sidebar.text_input("", "https://www.example.com/sample-image.jpg")

        # Get image from URL and predict
        if st.button("Predict"):
            with st.spinner("Fetching Image..."):
                try:
                    content = requests.get(image_path).content
                    st.image(content, use_column_width=True, caption="Original Image")
        
                    with st.spinner("Classifying..."):
                        img_tensor = load_image(content)
                        pred = model.predict(img_tensor)
                        pred_class = classes[np.argmax(pred)]
                        st.success(f"Predicted Class: {pred_class}")
        
                        # Grad-CAM Heatmap
                        st.write("Grad-CAM Heatmaps:")
                        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
                        # Original Image
                        axs[0].imshow(np.squeeze(img_tensor))
                        axs[0].set_title("Original Image")
                        axs[0].axis('off')
        
                        # Single Grad-CAM Heatmap
                        heatmap_single = make_gradcam_heatmap(img_tensor, model, "conv2d_3")
                        axs[1].imshow(heatmap_single)
                        axs[1].set_title("Single Grad-CAM Heatmap")
                        axs[1].axis('off')
        
                        st.pyplot(fig)
        
                        # Visualization for multiple convolutional layers
                        st.write("Grad-CAM Heatmaps for Convolutional Layers:")
                        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
                        layer_names = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3"]
        
                        for l, layer_name in enumerate(layer_names):
                            axs[l].imshow(make_gradcam_heatmap(img_tensor, model, layer_name))
                            axs[l].set_title(f"Convolutional Layer {l + 1}")
                            axs[l].axis('off')  # Turn off axes for each subplot
        
                        st.pyplot(fig)
        
                except requests.exceptions.RequestException:
                    st.error("Error fetching image. Please check the URL.")

    elif image_option == "Upload":
        # Upload image file
        st.sidebar.write("Choose an image to upload:")
        uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])
    
        if uploaded_file is not None:
            with st.spinner("Classifying..."):
                content = uploaded_file.read()
                st.image(content, use_column_width=True, caption="Original Image")
                img_tensor = load_image(content)
                pred = model.predict(img_tensor)
                pred_class = classes[np.argmax(pred)]
                st.success(f"Predicted Class: {pred_class}")

                # Grad-CAM Heatmap
                st.write("Grad-CAM Heatmaps:")
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                # Original Image
                axs[0].imshow(np.squeeze(img_tensor))
                axs[0].set_title("Original Image")
                axs[0].axis('off')

                # Single Grad-CAM Heatmap
                heatmap_single = make_gradcam_heatmap(img_tensor, model, "conv2d_3")
                axs[1].imshow(heatmap_single)
                axs[1].set_title("Single Grad-CAM Heatmap")
                axs[1].axis('off')

                st.pyplot(fig)

                # Visualization for multiple convolutional layers
                st.write("Grad-CAM Heatmaps for Convolutional Layers:")
                fig, axs = plt.subplots(1, 4, figsize=(24, 6))
                layer_names = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3"]

                for l, layer_name in enumerate(layer_names):
                    axs[l].imshow(make_gradcam_heatmap(img_tensor, model, layer_name))
                    axs[l].set_title(f"Convolutional Layer {l + 1}")
                    axs[l].axis('off')  # Turn off axes for each subplot

                st.pyplot(fig)
        
              
