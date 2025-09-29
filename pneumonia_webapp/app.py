# app.py

from flask import Flask, request, render_template, send_file, redirect, url_for, session
import os
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
from infer import predict_image  # ✅ Import from infer.py

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def generate_pdf(patient_data, predictions):
    report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'report.pdf')
    c = canvas.Canvas(report_path, pagesize=letter)

    c.setFont("Helvetica", 12)
    c.drawString(50, 750, f"Patient Name: {patient_data.get('name', '')}")
    c.drawString(50, 730, f"Age: {patient_data.get('age', '')}")
    c.drawString(50, 710, f"Gender: {patient_data.get('gender', '')}")
    c.drawString(50, 690, f"Symptoms: {patient_data.get('symptoms', '')}")
    c.drawString(50, 670, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = 640
    for pred in predictions:
        c.drawString(50, y, f"Image: {pred['filename']} - Label: {pred['label']} - Confidence: {pred['confidence']:.2f}")
        y -= 20
        if y < 50:
            c.showPage()
            y = 750

    c.save()
    return report_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        patient = {
            "name": request.form.get("patient_name", ""),
            "age": request.form.get("patient_age", ""),
            "gender": request.form.get("patient_gender", ""),
            "symptoms": request.form.get("patient_symptoms", "")
        }

        predictions = []
        files = request.files.getlist("xray_images")
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
                    "confidence": float(confidence)
                })

        generate_pdf(patient, predictions)

        session['predictions'] = predictions
        session['patient'] = patient

        return redirect(url_for('index'))

    predictions = session.get('predictions', [])
    patient = session.get('patient', {})
    return render_template("index.html", predictions=predictions, patient=patient)

@app.route("/download_report")
def download_report():
    report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'report.pdf')
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    return "Report not available.", 404

@app.route("/clear_session")
def clear_session():
    session.clear()
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)