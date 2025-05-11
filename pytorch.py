import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")

# Load and prepare an image
img_path = 'your_image.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict
preds = model.predict(x)
decoded = decode_predictions(preds, top=3)[0]

# Show results
print("Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded):
    print(f"{i+1}. {label}: {score:.2f}")