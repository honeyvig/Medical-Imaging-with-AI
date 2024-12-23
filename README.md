# Medical-Imaging-with-AI
The main goal of this project is the construction of a convolutional  network to classify SPECT images into its four stages of Parkinson's disease. I already processed the images that come from a tomography process and had to be 3D rendered and colored for it to be sliced into coronal photograms. Each stage has around 50 patients and each patient has around 30 images, except for the fourth stage which has no more than 15 patients data available at the foundation I am getting the data from. Referring to this issue, I asked my advisor about this lack of data and told me that it is no big issue since I can compensate for it with data from the other stages.
-------------------
To develop a convolutional neural network (CNN) to classify SPECT (Single Photon Emission Computed Tomography) images into the four stages of Parkinson's disease, we need to take several steps to ensure the proper processing and handling of the images and their data. Below is a general Python code outline for this type of project using deep learning techniques, which includes preprocessing the 3D SPECT images, splitting the data into training, validation, and testing sets, and defining and training the CNN.
Required Libraries:

    numpy: For handling arrays and numerical data.
    matplotlib: For visualizing data and images.
    tensorflow/keras: For building and training the CNN model.
    scikit-learn: For splitting the data and evaluating the model.
    SimpleITK: For loading and processing 3D medical images (if needed).
    cv2 or PIL: For basic image processing (resizing, augmentation).

You can install required libraries with:

pip install tensorflow scikit-learn SimpleITK matplotlib opencv-python

Code Outline:

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os

# Loading and Preprocessing SPECT images
def load_images(image_paths, label_mapping):
    images = []
    labels = []
    for image_path in image_paths:
        # Load the 3D image (assuming NIfTI format or similar)
        img = sitk.ReadImage(image_path)
        img_array = sitk.GetArrayFromImage(img)  # Convert to numpy array
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for 3D input
        
        # Normalize image
        img_array = (img_array - np.mean(img_array)) / np.std(img_array)

        # Get the label from the path (or another source of label data)
        label = label_mapping.get(os.path.basename(image_path), None)
        
        if label is not None:
            images.append(img_array)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Define CNN Model for 3D Image Classification
def build_cnn_model(input_shape):
    model = Sequential()

    # 3D Convolutional Layers
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Flatten and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))  # 4 output classes (4 stages of Parkinson's)

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Data Augmentation function (if needed)
def data_augmentation(images, labels):
    # Simple example of augmentation by flipping the images along different axes
    augmented_images = []
    augmented_labels = []
    
    for img, label in zip(images, labels):
        augmented_images.append(img)
        augmented_labels.append(label)
        
        # Flip image along axis (for example, flip on the X-axis)
        flipped_img = np.flip(img, axis=0)
        augmented_images.append(flipped_img)
        augmented_labels.append(label)
        
        # Flip along Y-axis
        flipped_img = np.flip(img, axis=1)
        augmented_images.append(flipped_img)
        augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)

# Main function to load data, build the model, and train it
def main():
    # Set paths and labels for images
    image_paths = [f"images/{file}" for file in os.listdir("images/") if file.endswith(".nii")]
    label_mapping = {
        'stage_1_patient_1.nii': 0,  # Stage 1: Parkinson's
        'stage_2_patient_1.nii': 1,  # Stage 2: Parkinson's
        'stage_3_patient_1.nii': 2,  # Stage 3: Parkinson's
        'stage_4_patient_1.nii': 3   # Stage 4: Parkinson's
    }
    
    # Load and preprocess the images and labels
    images, labels = load_images(image_paths, label_mapping)
    
    # Split the data into training, validation, and testing
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Augment the training data if needed
    X_train, y_train = data_augmentation(X_train, y_train)
    
    # Build the CNN model
    model = build_cnn_model(X_train.shape[1:])
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_val, y_val))

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc}")
    
    # Visualize predictions (optional)
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    
    plt.figure(figsize=(12, 8))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.imshow(X_test[i, ..., 0], cmap='gray')
        plt.title(f"True: {y_test[i]} Pred: {predicted_labels[i]}")
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()

Detailed Explanation:

    Image Preprocessing:
        The load_images function reads 3D medical images (assuming NIfTI format, but other formats could be used) and converts them into NumPy arrays for further processing.
        Images are normalized to ensure the network learns effectively.

    Data Augmentation:
        This helps address the imbalance in the dataset (especially the shortage of Stage 4 data). Basic augmentation strategies like flipping images on different axes help simulate additional data for training.

    CNN Model Architecture:
        The model uses 3D convolutional layers (Conv3D) to process the 3D SPECT images. These layers learn spatial hierarchies in the data.
        Max-pooling layers help reduce spatial dimensions to focus on important features.
        After the convolutional layers, the model flattens the output and adds dense layers to make final predictions.

    Training and Evaluation:
        The model is trained using the training set (X_train and y_train), and its performance is evaluated using a separate test set (X_test and y_test).
        The loss function used is sparse_categorical_crossentropy since we have multiple classes (stages of Parkinson’s).
        Accuracy is used as the evaluation metric.

    Visualization:
        You can visualize a few test samples and their predicted labels to ensure the model is making reasonable predictions.

Additional Notes:

    Data Imbalance: Since Stage 4 data is scarce, consider experimenting with techniques like synthetic data generation (using GANs or other advanced augmentation methods) or re-sampling strategies.
    Model Hyperparameters: Tune the batch size, epochs, and learning rate based on your dataset's size and computational resources.
    Model Optimization: Consider using transfer learning if there’s a pre-trained 3D model available that can be fine-tuned for your problem.

This approach should help you set up the framework for classifying SPECT images and dealing with the challenges posed by limited data for certain stages.
