import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

IMG_SIZE = (224, 224)

def preprocess_image_bytes(image_bytes):
    # image_bytes: raw bytes from uploaded file
    img = Image.open(image_bytes).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_model(model, image_array):
    pred = model.predict(image_array)[0][0]
    # pred: sigmoid value between 0 and 1
    return float(pred)
