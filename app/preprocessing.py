import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

def preprocess_image(image: Image.Image):
  

    image = image.resize((224, 224))

    x = img_to_array(image)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    return x