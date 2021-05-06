# Dependencies
from tensorflow.keras.models import model_from_json
import numpy as numpy
import tensorflow as tf 

# Set appropriate memory (optional)
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.comp.v1.session(config=config)

# Create the class that will hold the model
class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]