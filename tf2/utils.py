import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from IPython.display import display

import numpy as np
import pandas as pd


class MyResNet50:
    model = ResNet50(weights='imagenet')
    prediction_collector: dict = {}
    count_instances: int = 0

    def __init__(self, img_path: str):
        self.img_path: str = img_path
        self.img = self.preprocess_img()
        MyResNet50.count_instances += 1

    def decode_predict(self, display_res: bool = True):
        preds = MyResNet50.model.predict(self.img)
        decoded_predictions = decode_predictions(preds, top=5)[0]
        instance_number: int = MyResNet50.count_instances

        res: dict = {}
        for tup in decoded_predictions:
            _, breed, prob = tup
            res[breed] = prob

        MyResNet50.prediction_collector[instance_number] = res
        if display_res:
            print(res)
        return res

    def preprocess_img(self, display_img: bool = True):
        img = image.load_img(self.img_path, target_size=(224, 224))
        if display_img:
            display(img)
        img_array = image.img_to_array(img)
        img_expanded = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_expanded)

    @classmethod
    def get_frame(cls) -> pd.DataFrame:
        return pd.DataFrame.from_dict(cls.prediction_collector, orient='index')

    @classmethod
    def sort_freq(cls) -> pd.DataFrame:
        df = cls.get_frame()
        m = np.zeros_like(df.values)
        m[np.arange(len(df)), np.nanargmax(df.values, axis=1)] = 1
        df1 = pd.DataFrame(m, columns=df.columns).astype(int)
        return df1.sum().sort_values(ascending=False)

    @classmethod
    def sort_prob(cls):
        df = cls.get_frame()
        return df.sum(axis=0).sort_values(ascending=False)



