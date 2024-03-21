import streamlit as st
import numpy as np
from PIL import Image
import torch
from mtcnn.mtcnn import MTCNN
import time
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from FakeFaceDetection_prediction import predict
from objectdetection import compare_models

def process_file(uploaded_file):
    image = Image.open(uploaded_file)
    image = image.resize((256, 256))
    return np.array(image)


@st.cache_resource(ttl=None)
def loadmodels():
    discrimator = load_model("artifacts/discriminator.h5")
    generator = load_model("artifacts/generator.h5")

    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    realvfake = load_model("artifacts/realvfake.h5")

    mtcnn = MTCNN()
    return discrimator, generator, yolo, realvfake,mtcnn

def check_human(yolo,img,mtcnn):
    mtcnn_acc, yolo_acc,model_name = compare_models(pixels=img)
    if mtcnn_acc>0.0 and yolo_acc>0.0:
        st.write(f"Using {model_name} for face detection")
        return True
def plot_generated_images(mtcnn, realfake, yolo,generator,img, square = 5, epochs = 0, latent_dim = 100):
    if check_human(yolo=yolo,mtcnn=mtcnn,img=img):
        fig = plt.figure(figsize = (10,10))
        for i in range(square * square):
            if epochs != 0:    
                if(i == square //2):
                    plt.title("Generated Image at Epoch:{}\n".format(epochs), fontsize = 32, color = 'black')
            plt.subplot(square, square, i+1)
            noise = np.random.normal(0,1,(1,latent_dim))
            img = generator(noise)
            img_np = np.array(img[0,...]+1)
            label = predict(img_np,model=realfake)
            plt.title(label=label, fontsize = 16, color = 'black')
            plt.imshow(np.clip((img[0,...]+1)/2, 0, 1))
        return st.pyplot(fig)
    else:
        return st.write("No person detected in the image")
    
def main():
    discrimator, generator, yolo, realvfake,mtcnn = loadmodels()
    st.title("Image Uploader")
    st.write("Please upload an image file.")
    option = st.selectbox("Select an option:", ("Take a photo", "Try a Demo"))

    if option == "Take a photo":
        image = st.camera_input("Capture image")
        if image is not None:
            image = process_file(image)
            plot_generated_images(mtcnn=mtcnn,realfake=realvfake,yolo=yolo,generator=generator,img = image, square=2)

    elif option == 'Try a Demo':
        image = np.array(Image.open("artifacts/demo.webp"))
        st.image(image)

        st.write("#### Generated images")
        plot_generated_images(mtcnn=mtcnn,realfake=realvfake,yolo=yolo,generator=generator,img = image, square=2)
    
if __name__ == "__main__":
    main()