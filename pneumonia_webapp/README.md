Pneumonia Detection Web App
An AI-powered diagnostic tool that detects pneumonia from chest X-ray images using a deep learning model built on MobileNetV2. The app provides a clean web interface for uploading images, entering patient data, viewing predictions, and downloading PDF reports.

 Features
- Upload multiple chest X-ray images
- Predict pneumonia vs. normal using a trained CNN model
- Display confidence scores and simulated change metrics
- Input patient details (name, age, gender, symptoms)
- Download a professional PDF report with embedded images
- Clean UI built with Flask and Bootstrap
- Chart.js integration for visualizing prediction trends

Tech Stack
    Layer           Tools Used
    Frontend         HTML, CSS, Bootstrap Chart.js
   Backend          Flask, TensorFlow/Keras, NumPy
   Model           MobileNetV2 (transfer learning)
   PDF Reports      ReportLab
   Deployment      Localhost/Render/ 



Folder Structur

pneumonia_webapp/
├── app.py
├── templates/
│   └── index.html
├── static/
│   ├── uploads/
│   ├── css/
│   └── js/
├── model/
│   └── pneumonia_classifier_model.keras
├── data/
│   ├── train/
│   └── val/
 Model Training (Optional)
If retraining is needed, use train_model.py with your dataset structured as:
data/
├── train/
│   ├── PNEUMONIA/
│   └── NORMAL/
├── val/
│   ├── PNEUMONIA/
│   └── NORMAL/

Model is trained using MobileNetV2 with class weighting and AUC metrics.

⚙️ Setup Instructions
- Clone the repo
- Install dependencies:

pip install -r requirements.txt

- Place your trained model in model/pneumonia_classifier_model.keras
- Run the app:
python app.py

- Open in browser:

http://127.0.0.1:5000
 Sample PDF Report
Each report includes:
- Patient details
- Prediction results
- Confidence scores
- Embedded image previews
- Timestamp and layout for clinical use

🛡️ Ethical Considerations
This tool is intended for educational and research purposes. It should not be used for real-world diagnosis without proper validation and regulatory approval.

👤 Author
Fenet — Full-stack developer focused on ethical AI, clean architecture, and empowering others through technology.
