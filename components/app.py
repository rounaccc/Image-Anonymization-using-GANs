import streamlit as st
import numpy as np
from PIL import Image
import torch
from mtcnn.mtcnn import MTCNN
import time
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from objectdetection import compare_models
from FakeFaceDetection_prediction import make_prediction

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

def create_imgs(generator,square,latent_dim = 100):
    for i in range(square*square):
        noise = np.random.normal(0,1,(1,latent_dim))
        img = generator(noise)
        plt.imsave(f"image_gen/gen_{i}.png", np.clip((img[0,...]+1)/2, 0, 1))

def check_realvfake(realvfake,img):
    prediction_label,score = make_prediction(realvfake, image_path=img)
    return prediction_label,score

def plot_generated_images(realvfake,mtcnn, yolo,generator,img, square = 5, epochs = 0, latent_dim = 100):
    if check_human(yolo=yolo,mtcnn=mtcnn,img=img):
        create_imgs(generator,square,latent_dim)
        fig = plt.figure(figsize = (10,10))
        for i in range(square * square):
            if epochs != 0:    
                if(i == square //2):
                    plt.title("Generated Image at Epoch:{}\n".format(epochs), fontsize = 32, color = 'black')
            plt.subplot(square, square, i+1)
            img_path = f"image_gen/gen_{i}.png"
            realvfake_pred,score = check_realvfake(realvfake, img_path)
            plt.imshow(plt.imread(img_path))
            plt.title(f"{realvfake_pred}, {round(score[0][0]*100,2)}%", fontsize = 10, color = 'black')
        st.pyplot(fig)
        return True
    else:
        st.write("No person detected in the image")
        return False
    
def main():
    discrimator, generator, yolo, realvfake,mtcnn = loadmodels()
    st.title("Image Uploader")
    st.write("Please upload an image file.")
    option = st.selectbox("Select an option:", ("Take a photo", "Try a Demo"))

    if option == "Take a photo":
        image = st.camera_input("Capture image")
        if image is not None:
            image = process_file(image)
            flag = plot_generated_images(realvfake=realvfake,mtcnn=mtcnn,yolo=yolo,generator=generator,img = image, square=2)

    elif option == 'Try a Demo':
        image = np.array(Image.open("artifacts/demo.webp"))
        st.image(image)

        st.write("#### Generated images")
        flag = plot_generated_images(realvfake=realvfake,mtcnn=mtcnn,yolo=yolo,generator=generator,img = image, square=2)
    
if __name__ == "__main__":
    main()