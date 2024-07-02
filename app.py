from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import requests
import joblib
import os
import logging
import base64
import time
from threading import Thread

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load model once during startup
knn_model = joblib.load('knn_model.pkl')

# Global list to store all fetched Firebase data
all_data = []

def remove_background_with_rembg(image_path):
    logging.debug(f"Removing background for {image_path}")
    original_image = cv2.imread(image_path)
    resized_image = cv2.resize(original_image, (1080, 1080))
    cv2.imwrite('static/uploads/resized_image.png', resized_image)
    input_image = Image.open('static/uploads/resized_image.png')
    output_image = remove(input_image)
    output_path = 'static/processed/processed_image.png'
    output_image.save(output_path)
    processed_image = cv2.imread(output_path)
    logging.debug(f"Background removed for {image_path}")
    return processed_image

def extract_features(image_path):
    logging.debug(f"Extracting features for {image_path}")
    preprocessed_image = remove_background_with_rembg(image_path)
    hsv_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv_image, axis=(0, 1))
    mean_rgb = np.mean(preprocessed_image, axis=(0, 1))
    features = np.concatenate([mean_hsv, mean_rgb])
    logging.debug(f"Features extracted for {image_path}")
    return features, preprocessed_image

def process_prediction(image_path):
    logging.debug(f"Processing prediction for {image_path}")
    image_features, preprocessed_image = extract_features(image_path)
    predicted_label = knn_model.predict([image_features])
    logging.debug(f"Prediction processed for {image_path}")
    return predicted_label[0], preprocessed_image

def check_for_nutrient_deficiency(data):
    logging.debug("Checking for nutrient deficiency")
    npk_value = data.get('Nitrogen', 0) + data.get('Phosphorous', 0) + data.get('Potassium', 0)
    if npk_value > 50 and data.get('ph', 0) < 7:
        return "Kekurangan Unsur Hara"
    else:
        return "Tidak ada kekurangan unsur hara"

def fetch_firebase_data():
    logging.debug("Fetching Firebase data")
    url = "https://tugasakhir-189d6-default-rtdb.asia-southeast1.firebasedatabase.app/data1.json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad response
        data = response.json()
        logging.debug("Firebase data fetched")
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Firebase data: {str(e)}")
        return {}

def continuous_fetch():
    global all_data
    # Fetch initial data
    data = fetch_firebase_data()
    if data:
        all_data.append({'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'data': data})
    while True:
        try:
            data = fetch_firebase_data()
            if data:
                all_data.append({'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'data': data})
            time.sleep(2)
        except Exception as e:
            logging.error(f"Error fetching Firebase data in continuous fetch: {str(e)}")
            time.sleep(2)

# Start continuous fetching in a separate thread
Thread(target=continuous_fetch, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/firebase_data.html')
def firebase_data_view():
    return render_template('firebase_data.html', all_data=all_data)

# Endpoint untuk reset data
@app.route('/reset_data', methods=['POST'])
def reset_data():
    global all_data
    all_data = []
    return 'Data reset successful'

@app.route('/upload', methods=['POST'])
def upload_file():
    logging.debug("File upload requested")
    if 'file' not in request.files and 'image_data' not in request.form:
        logging.error("No file part in the request")
        return redirect(request.url)

    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            logging.error("No selected file")
            return redirect(request.url)
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
        logging.debug(f"File saved to {file_path}")
    else:
        image_data = request.form['image_data']
        file_path = os.path.join('static/uploads', 'captured_image.png')
        with open(file_path, "wb") as fh:
            fh.write(base64.b64decode(image_data.split(',')[1]))
        logging.debug(f"Image captured and saved to {file_path}")
    
    try:
        predicted_label, processed_image = process_prediction(file_path)
        logging.debug(f"Predicted label: {predicted_label}")
        data = fetch_firebase_data()
        nutrient_status = check_for_nutrient_deficiency(data)
        processed_image_path = 'static/processed/processed_image.png'
        return render_template('result.html', 
                               original_image=file_path, 
                               processed_image=processed_image_path, 
                               predicted_label=predicted_label, 
                               nutrient_status=nutrient_status,
                               firebase_data=data)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return redirect(url_for('index'))

@app.route('/fetch_latest_data')
def fetch_latest_data():
    try:
        data = fetch_firebase_data()
        return jsonify(data)
    except Exception as e:
        logging.error(f"Error fetching Firebase data: {str(e)}")
        return jsonify({})

@app.route('/fetch_all_data')
def fetch_all_data():
    return jsonify(all_data)

#if __name__ == "__main__":
   # app.run(debug=True)
