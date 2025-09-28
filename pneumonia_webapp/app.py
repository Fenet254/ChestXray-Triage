from flask import Flask, request, render_template, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model("model/pneumonia_classifier_model.keras")

# Prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    confidence = float(round(prediction * 100, 2)) if prediction > 0.5 else float(round((1 - prediction) * 100, 2))
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"

    return label, confidence

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    predictions = []
    patient_data = {}

    if request.method == "POST":
        files = request.files.getlist("xray_images")
        patient_data["name"] = request.form["patient_name"]
        patient_data["age"] = request.form["patient_age"]
        patient_data["gender"] = request.form["patient_gender"]
        patient_data["symptoms"] = request.form["patient_symptoms"]

        for file in files:
            if file:
                filename = f"{uuid.uuid4().hex}_{file.filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                label, confidence = predict_image(filepath)
                predictions.append({
                    "filename": filename,
                    "filepath": filepath,
                    "label": label,
                    "confidence": confidence,
                    "change": round(np.random.uniform(0.5, 5.0), 2)
                })

    return render_template("index.html", predictions=predictions, patient=patient_data)

# PDF generation
def generate_pdf(patient_data, predictions, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Pneumonia Detection Report")
    y -= 30

    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20
    c.drawString(50, y, f"Name: {patient_data['name']}")
    y -= 20
    c.drawString(50, y, f"Age: {patient_data['age']}")
    y -= 20
    c.drawString(50, y, f"Gender: {patient_data['gender']}")
    y -= 20
    c.drawString(50, y, f"Symptoms: {patient_data['symptoms']}")
    y -= 40

    for i, result in enumerate(predictions):
        if y < 150:
            c.showPage()
            y = height - 50

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, f"Image {i+1}: {result['filename']}")
        y -= 20
        c.setFont("Helvetica", 12)
        c.drawString(50, y, f"Prediction: {result['label']} ({result['confidence']}% confidence)")
        y -= 20

        try:
            c.drawImage(result['filepath'], 50, y - 150, width=200, height=150)
            y -= 170
        except:
            c.drawString(50, y, "Image preview unavailable.")
            y -= 30

    c.save()

# PDF download route
@app.route("/download_report", methods=["POST"])
def download_report():
    patient_data = {
    "name": request.form.get("patient_name", "Unknown"),
    "age": request.form.get("patient_age", "N/A"),
    "gender": request.form.get("patient_gender", "N/A"),
    "symptoms": request.form.get("patient_symptoms", "N/A")
}
    predictions = eval(request.form["predictions"])  # Use safer serialization in production

    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_report.pdf")
    generate_pdf(patient_data, predictions, pdf_path)
    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)