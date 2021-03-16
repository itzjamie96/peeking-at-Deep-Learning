import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.models import load_model
from PIL import Image
import tensorflow as tf


# Code for Correcting Error
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Set Basic Parameters
# # Prediction Image Path
image_path = "./example.jpg"
# # Trained Model Path (.h5 file)
model_path = "./fish_classification.h5"
# # Size of Images
img_height = 128
img_width = 128


# Transform Image for Prediction
def Dataization(image_path, img_w, img_h):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize((img_w, img_h))
    img_data = np.asarray(img)
    return img_data/255


# Set Predict Condition
# # Set Image
pred_data = [Dataization(image_path, img_width, img_height)]
pred_data = np.array(pred_data)
# # Load Model (.h5 file)
pred_model = load_model(model_path)
# # Predict and Save Result
pred_result = pred_model.predict(pred_data)

print("Hello")