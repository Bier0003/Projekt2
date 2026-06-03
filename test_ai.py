
from PIL import Image  
import numpy as np 
import tensorflow as tf


img = Image.open('uploads/test_pic.png')


img_resized = img.resize((224, 224))

print("resized picture size: ", img_resized.size)

img_array = np.array(img_resized)

img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

img_array = np.expand_dims(img_array, axis=0)

print(" loading AI model...")
model = tf.keras.applications.MobileNetV2(weights='imagenet')

print (" AI model loaded, making prediction...")
predictions = model.predict(img_array)

results = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]

print(f"Predicted label: {results[0][1]} with confidence {results[0][2]:.2f}%)")