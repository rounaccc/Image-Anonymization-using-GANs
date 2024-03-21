import streamlit as st
import numpy as np
from PIL import Image
import torch
import time
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from FakeFaceDetection_prediction import predict

def process_file(uploaded_file):
    image = Image.open(uploaded_file)
    image = image.resize((256, 256))
    return np.array(image)


@st.cache_resource(ttl=None)
def loadmodels():
    discrimator = load_model("artifacts/discriminator.h5")
    generator = load_model("artifacts/generator.h5")

    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    realvfake = load_model("artifacts/realfake.h5")


    return discrimator, generator, yolo, realvfake

def check_human(model,img):
    results = model(img)
    for obj in results.xyxy[0]:
        confidence = obj[4].item() * 100
        label = model.names[int(obj[5])]
        print(label, confidence, label == "person" and confidence > 50)
        if label == "person" and confidence > 50:
            return True
    return False
def plot_generated_images(realfake, yolo,generator,img, square = 5, epochs = 0, latent_dim = 100):
    if check_human(yolo,img=img):
        fig = plt.figure(figsize = (10,10))
        for i in range(square * square):
            if epochs != 0:    
                if(i == square //2):
                    plt.title("Generated Image at Epoch:{}\n".format(epochs), fontsize = 32, color = 'black')
            plt.subplot(square, square, i+1)
            noise = np.random.normal(0,1,(1,latent_dim))
            img = generator(noise)
            img_np = np.array(img[0,...])
            label = predict(img_np,model=realfake)
            plt.title(label=label, fontsize = 16, color = 'black')
            plt.imshow(np.clip((img[0,...]+1)/2, 0, 1))
        return st.pyplot(fig)
    else:
        return st.write("No person detected in the image")
    
def main():
    discrimator, generator, yolo, realvfake = loadmodels()
    st.title("Image Uploader")
    st.write("Please upload an image file.")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    option = st.selectbox("Select an option:", ("Take a photo", "Upload an image", "Try a Demo"))

    if option == "Take a photo":
        image = st.camera_input("Capture image")
        if image is not None:
            image = process_file(image)
            plot_generated_images(realfake=realvfake,yolo=yolo,generator=generator,img = image, square=2)

    elif option == "Upload an image":
        uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png", "jpeg", "heic", "webp"])
        if uploaded_file is not None:
            image = process_file(uploaded_file)
            st.image(image)
            plot_generated_images(realfake=realvfake,yolo=yolo,generator=generator,img = image, square=2)

    elif option == 'Try a Demo':
        image = np.array(Image.open("paudhayodha/assets/apple_scab.jpeg"))
        st.image(image)

        st.write("Demo image: Apple with Scab")
        with st.spinner('loading prediction'):
            time.sleep(0.8)

        st.write("#### Prediction:")
        plot_generated_images(realfake=realvfake,yolo=yolo,generator=generator,img = image, square=2)


    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
if __name__ == "__main__":
    main()