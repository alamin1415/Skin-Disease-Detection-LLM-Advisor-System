import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

MODEL_PATH = "model/model_vgg16.h5"

class_names = {
    0: "eczema",
    1: "warts",
    2: "melanoma",
    3: "atopic dermatitis",
    4: "bcc",
    5: "nevus",
    6: "bkl",
    7: "psoriasis",
    8: "seborrheic",
    9: "tinea"
}

def load_skin_model():
    model = load_model(MODEL_PATH)
    return model, class_names


def predict_skin(model, image, class_names):
    image = image.resize((224, 224))

    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)

    pred_class = np.argmax(preds)
    confidence = np.max(preds)

    return class_names[pred_class], confidence