'''
@author: Tanisha Mangliya
Email: tanishamangliya@gmail.com
Date: 12-08-2025
'''

from utils import data_manager as dm
from utils.config import configureData
from utils.config import configureModel
import tensorflow as tf
import os
import numpy as np


config_data = configureData()
config_model = configureModel()

config_data['PREDICTION_DATA_DIR'] = r"D:\\Image-Classify-UI(CatDog)\\prediction"

print("config_data keys:", config_data.keys())
print("PREDICTION_DATA_DIR in config_data?", 'PREDICTION_DATA_DIR' in config_data)

config_data['PREDICTION_DATA_DIR'] = r"D:\\Image-Classify-UI(CatDog)\\prediction"


#Manage Image
image_list = os.listdir(config_data['PREDICTION_DATA_DIR'])



def predict():

    """The logic for prediction step.
  
    This method should contain the mathematical logic prediction.
    This typically includes the forward pass with respect to updated weights.

     Args:
      data: A nested structure of `Tensor`s.

    Returns:
      A `nd array` containing values.Typically, the
      values of the `Model`'s metrics are returned. Example:
      `[[0,1]]`.

    """

    # load model
    model_path = f"New_trained_model/{'new' + config_model['MODEL_NAME'] + '.h5'}"
    model = tf.keras.models.load_model(model_path)
    for image in image_list:
        predict = dm.manage_input_data(os.path.join(config_data['PREDICTION_DATA_DIR'],image))

        print("Loaded data shape:", None if predict is None else predict.shape)
         

        result = model.predict(predict)
        results = np.argmax(result, axis=-1)
        print(f"Original image : {image}. Predicted as {results}")


