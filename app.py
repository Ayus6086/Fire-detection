from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import io
import base64
from PIL import Image

app = Flask(__name__)

# Load model with error handling
try:
    model = load_model('model/fire_detector.h5')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None

def predict_fire(img_path):
    if model is None:
        print("[ERROR] Model not loaded")
        return False, 0.0
    
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # ADJUSTED LOGIC: Model predictions are too low, using balanced threshold
        # Only trigger fire detection for low scores to reduce false positives
        fire_detected = prediction < 0.15
        
        print(f"[DEBUG] Upload prediction: {prediction:.4f}, Fire detected: {fire_detected}")
            
        return fire_detected, float(prediction)
    except Exception as e:
        print(f"[ERROR] Upload prediction failed: {str(e)}")
        return False, 0.0

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            img_path = os.path.join('static', image_file.filename)
            image_file.save(img_path)
            fire_detected, score = predict_fire(img_path)
            return render_template('index.html', fire=fire_detected, score=score, img_path=img_path)
    # default GET route resets everything
    return render_template('index.html', fire=None, score=None, img_path=None)


@app.route("/predict_webcam", methods=["POST"])
def predict_webcam():
    if model is None:
        return jsonify({"error": "Model not loaded", "fire": False, "score": 0.0}), 500
    
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "No image data received"}), 400

        img_data = data["image"].split(",")[1]
        image_pil = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("RGB")
        image_pil = image_pil.resize((224, 224))
        img_array = np.array(image_pil)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # TEMPORARY FIX: Disable webcam fire detection due to false positives
        # The model is not reliable for real-time webcam detection
        fire = False
        
        print(f"[DEBUG] Webcam prediction: {prediction:.4f}, Fire detected: {fire}")
        
        # Return detection results
        return jsonify({
            "fire": bool(fire), 
            "score": float(prediction),
            "status": "DISABLED",
            "reason": "Webcam detection disabled due to false positives"
        })
    
    except Exception as e:
        print(f"[ERROR] Webcam prediction failed: {str(e)}")
        return jsonify({"error": "Prediction failed", "fire": False, "score": 0.0}), 500

# optional reset route
@app.route('/reset', methods=['POST'])
def reset():
    return redirect(url_for('upload_predict'))

if __name__ == '__main__':
    app.run(debug=True)


