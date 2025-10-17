import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model once
model = load_model("model/pneumonia_classifier_model.keras")

def predict_image(filepath):
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]  # Binary output
    label = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = prediction if label == "Pneumonia" else 1 - prediction
    return label, float(confidence)