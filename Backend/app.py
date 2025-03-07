from flask import Flask, request, jsonify
import cv2
import numpy as np
import pywt
import joblib
import json

app = Flask(__name__)

# Load trained model and label encoder
model_path = 'Model/saved_model.pkl'  
label_encoder_path = 'Model/label_encoder.pkl'
celebrities_json = 'Model/class_dict.json'

model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

# Load celebrity mapping (label to name mapping)
with open(celebrities_json, 'r') as f:
    label_to_name = json.load(f)

# Haar cascades for face & eye detection (same as training pipeline)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

target_size = (64, 64)
wavelet = 'db1'


def get_cropped_face_with_2_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = img[y:y + h, x:x + w]
        gray_face_roi = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(gray_face_roi)

        if len(eyes) >= 2:
            return face_roi

    return None


def extract_wavelet_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, target_size) / 255.0

    coeffs = pywt.dwt2(img_resized, wavelet)
    cA, (cH, cV, cD) = coeffs

    raw_flat = img_resized.flatten()
    cA_flat = cA.flatten()
    cH_flat = cH.flatten()
    cV_flat = cV.flatten()
    cD_flat = cD.flatten()

    stacked_features = np.hstack((raw_flat, cA_flat, cH_flat, cV_flat, cD_flat))
    return stacked_features.reshape(1, -1)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_bytes = file.read()
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    cropped_face = get_cropped_face_with_2_eyes(img)
    if cropped_face is None:
        return jsonify({'error': 'Could not detect a face with exactly 2 eyes'}), 400

    features = extract_wavelet_features(cropped_face)

    # Get prediction probabilities for all classes
    probabilities = model.predict_proba(features)[0]

    # Pair each label index with its probability
    label_probabilities = [(index, prob) for index, prob in enumerate(probabilities)]

    # Sort by probability (descending)
    label_probabilities = sorted(label_probabilities, key=lambda x: x[1], reverse=True)

    # Top 5 most similar celebrities
    top5 = label_probabilities[:5]

    # Convert to {name: similarity} using label_to_name mapping
    similar_celebrities = []
    for label, similarity in top5:
        celebrity_name = list(label_to_name.keys())[list(label_to_name.values()).index(label)]
        similar_celebrities.append({
            "name": celebrity_name,
            "similarity": round(float(similarity), 3)
        })

    top_celebrity = similar_celebrities[0]["name"]

    return jsonify({
        "top_celebrity": top_celebrity,
        "similar_celebrities": similar_celebrities
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
