from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os 
import cv2
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

label_mapping = {
    0: 'FMD',
    1: 'FMD',
    2: 'FMD',
    3: 'FMD',
    4: 'Healthy',
    5: 'Healthy',
    6: 'Healthy',
    7: 'Healthy'
}

@app.route('/')
def index():
    return render_template('start.html')

@app.route('/handle_signup', methods=['POST'])
def signup_post():
    return render_template('signup.html')

@app.route('/handle_login', methods=['POST'])
def login_post():
    return render_template('login.html')

@app.route('/main', methods=['POST'])
def enter():
    return render_template('main.html')

@app.route('/result', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print("File path:", filepath)

        model = load_model('vgg16_final_model.h5')

        img = cv2.imread(filepath)
        img = cv2.resize(img, (150, 150)) 
        img_normalized = img / 255.0 
        img_normalized = np.expand_dims(img_normalized, axis=0) 

        prediction = model.predict(img_normalized)
        prediction_index = np.argmax(prediction)
        prediction_label = label_mapping[prediction_index]

        return render_template('main.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
