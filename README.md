# 🔥 AI-Powered Fire Detection System

An AI-based fire detection web application built using **Flask** and **TensorFlow/Keras** that detects the presence of fire in images using a deep learning model.  
The system supports **image upload** and **webcam input** (currently disabled due to accuracy concerns) and displays prediction results with confidence scores.

---

## 🚀 Features

- 🔥 Fire detection using a trained deep learning model  
- 🖼️ Image upload and prediction via web interface  
- 📊 Confidence score display for each prediction  
- 🌐 REST API endpoint for webcam-based prediction  
- ⚠️ False-positive control using calibrated thresholds  
- 🧠 Robust error handling for model and input failures  

---

## 🧠 How It Works

1. User uploads an image through the web interface  
2. The image is resized to **224 × 224** and normalized  
3. The preprocessed image is passed to the trained CNN model  
4. The model outputs a probability score  
5. Based on a threshold, fire presence is determined  
6. The result and confidence score are shown on the UI  

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)  
- **Deep Learning:** TensorFlow, Keras  
- **Frontend:** HTML, CSS, JavaScript  
- **Image Processing:** Pillow (PIL), NumPy  

---
