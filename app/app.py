import os
import numpy as np
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

app = Flask(__name__)

# ==================== PATH SETUP ====================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))          # app folder
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))   # poultry-vision root
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'poultry_disease_classifier_final.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==================== LOAD MODEL ====================
print("Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ==================== CLASS & TREATMENT ====================
class_names = ['Coccidiosis', 'Healthy', 'Newcastle Disease', 'Salmonella']

treatments = {
    'Coccidiosis': 'Use anticoccidial drugs (e.g., Amprolium or Sulfonamides). Isolate affected birds and improve sanitation.',
    'Healthy': 'Bird appears healthy. Continue good hygiene, balanced diet, and regular monitoring.',
    'Newcastle Disease': 'No treatment available. Immediate quarantine required. Report to veterinary authorities.',
    'Salmonella': 'Use antibiotics (e.g., Enrofloxacin) under vet supervision. Improve hygiene and water quality.'
}

# ==================== IMAGE PREPROCESSING ====================
def prepare_image(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype('float32'))
    return img_array

# ==================== ROUTES ====================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file selected")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            img_array = prepare_image(filepath)
            predictions = model.predict(img_array)
            pred_idx = np.argmax(predictions[0])
            disease = class_names[pred_idx]
            confidence = round(predictions[0][pred_idx] * 100, 2)

            return render_template('result.html',
                                   prediction=disease,
                                   confidence=confidence,
                                   treatment=treatments[disease],
                                   image_file=filename)
        except Exception as e:
            return render_template('index.html', error=f"Prediction error: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)