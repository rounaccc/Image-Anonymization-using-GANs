import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


#function for preprocessing the input image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

#function for making predictions
def make_prediction(model, image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    if prediction > 0.5:
        return "Real"
    else:
        return "Fake"

def main():
    # Load the pre-trained model
    model = load_model('C:/Users/Rounak/Desktop/OneDrive/College/Projects/Image Anonymization/artifacts/Real VS Fake.h5')  # Change the filename if necessary
    # Take user input for the image path
    #image_path = input("Enter the path to the image: ")
    image_path = 'C:/Users/Rounak/Desktop/OneDrive/College/Projects/Image Anonymization/artifacts/demo.webp'

    # Check if the image path is valid
    if not os.path.exists(image_path):
        print("Invalid image path. Please provide a valid path.")
        return

    # Make prediction
    prediction = make_prediction(model, image_path)

    # Provide output
    print("Prediction:", prediction)

if __name__ == "__main__":
    main()