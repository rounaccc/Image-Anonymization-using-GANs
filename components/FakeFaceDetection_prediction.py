import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(image):
  # image = np.array(image)
  img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  img = cv2.resize(img, (224, 224))
  img = img.astype("float32") / 255.0
  img = np.expand_dims(img, axis=0)
  print(img.shape,"preprocessed")
  return img

def get_user_input():
  image_path = input("Enter the path to the image: ")
  return image_path

def load_my_model():
  
  model = load_model("C:/Users/Hetvi/Desktop/OneDrive/Projects/mloa nndl/Real VS Fake.h5")
  return model

def predict(image, model):

  preprocessed_image = preprocess_image(image)
  print(preprocessed_image[0].shape)
  prediction = model.predict(preprocessed_image[0])[0][0]
  class_label = "real" if prediction > 0.5 else "fake"
  return class_label

if __name__ == "__main__":

  image_path = get_user_input()
  model = load_my_model()
  prediction = predict(image_path, model)
  print(f"Predicted class: {prediction}")
