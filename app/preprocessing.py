import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

def preprocess_image(image: Image.Image):
    """
    Convert uploaded image into model-ready format for VGG16
    """

    # Step 1: Resize image to 224x224 (VGG16 input size)
    image = image.resize((224, 224))

    # Step 2: Convert PIL image to numpy array
    x = img_to_array(image)

    # Step 3: Add batch dimension (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)

    # Step 4: Apply VGG16 preprocessing (normalization)
    x = preprocess_input(x)

    return x