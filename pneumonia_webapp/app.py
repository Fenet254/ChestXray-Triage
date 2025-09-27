from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model/pneumonia_classifier_model.keras")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "PNEUMONIA" if prediction > 0.5 else "NORMAL"

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        file = request.files["xray_image"]
        filepath = os.path.join("static/uploads", file.filename)
        file.save(filepath)
        result = predict_image(filepath)
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)