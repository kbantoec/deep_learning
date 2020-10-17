from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

import numpy as np


def get_result_resnet50(img_path: str) -> list:
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expanded = np.expand_dims(img_array, axis=0)
    img_ready = preprocess_input(img_expanded)
    model = ResNet50(weights='imagenet')
    preds = model.predict(img_ready)
    return decode_predictions(preds)
