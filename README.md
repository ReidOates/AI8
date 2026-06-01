# 🚦 Traffic Sign Recognition using Random Forest (Flask Web App)

## 📌 Project Overview

This project is a machine learning-based web application that classifies traffic signs using the Random Forest algorithm combined with HOG (Histogram of Oriented Gradients) feature extraction.

The system allows users to upload an image and receive real-time classification results through a Flask-based web interface.

---

## 🎯 Objectives

- Build a machine learning model for image classification
- Apply HOG feature extraction for image representation
- Train Random Forest classifier on GTSRB dataset
- Deploy model into a Flask web application
- Provide interactive UI for prediction

---

## 📊 Dataset

The dataset used in this project is the **German Traffic Sign Recognition Benchmark (GTSRB)**.

Dataset source:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

- 43 traffic sign classes
- ~39,000 training images
- Preprocessed into grayscale and resized images

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Resize images to 48×48 pixels
- Convert to grayscale
- Extract features using HOG

### 2. Model Training
- Algorithm: Random Forest Classifier
- n_estimators: 50
- random_state: 42
- Training performed using extracted HOG features

### 3. Evaluation
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

---

## 📈 Results

- Accuracy: ~93%
- Model size: ~149 MB
- Good balance between performance and efficiency

---

## 🧠 Technologies Used

- Python
- Flask
- Scikit-learn
- OpenCV
- Scikit-image (HOG)
- NumPy
- HTML, CSS (Bootstrap)

---

## 🖥️ Features

- Upload traffic sign image
- Real-time prediction
- Confidence score display
- Simple and responsive UI

---

## 🚀 How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run application
python app.py

### 3. Open browser
http://127.0.0.1:5000

---

## 📁 Project Structure

ai8/
├── app.py
├── random_forest_gtsrb.pkl
├── requirements.txt
├── README.md
├── static/
│   ├── css/
│   └── uploads/
├── templates/
│   ├── index.html
│   └── result.html

---

## 📌 Notes

- Dataset is not included in this repository due to large size
- Model is pre-trained and saved as .pkl file
- This project is intended for academic purposes

---

## 👨‍💻 Author

Student project – Machine Learning with Flask Integration
