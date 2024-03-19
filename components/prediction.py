import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load generator model
generator = load_model('./artifacts/generator.h5')

def plot_generated_images(square = 5, epochs = 0, latent_dim = 100):
  plt.figure(figsize = (10,10))
  for i in range(square * square):
    if epochs != 0:    
        if(i == square //2):
            plt.title("Generated Image at Epoch:{}\n".format(epochs), fontsize = 32, color = 'black')
    plt.subplot(square, square, i+1)
    noise = np.random.normal(0,1,(1,latent_dim))
    img = generator(noise)
    plt.imshow(np.clip((img[0,...]+1)/2, 0, 1))
    
    plt.xticks([])
    plt.yticks([])
    plt.grid()

def main():
    prediction_grid_size = 2
    plot_generated_images(prediction_grid_size)
    plt.show()

if __name__ == "__main__":
    main()