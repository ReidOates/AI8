from flask import Flask, render_template, request
import cv2
import joblib
import numpy as np
import os

from skimage.feature import hog

app = Flask(__name__)

# Load model
model = joblib.load("random_forest_gtsrb.pkl")

# Folder upload
UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Nama kelas GTSRB
class_names = {
    0:"Speed Limit 20 km/h",
    1:"Speed Limit 30 km/h",
    2:"Speed Limit 50 km/h",
    3:"Speed Limit 60 km/h",
    4:"Speed Limit 70 km/h",
    5:"Speed Limit 80 km/h",
    6:"End Speed Limit 80 km/h",
    7:"Speed Limit 100 km/h",
    8:"Speed Limit 120 km/h",
    9:"No Passing",
    10:"No Passing Trucks",
    11:"Right-of-Way",
    12:"Priority Road",
    13:"Yield",
    14:"Stop",
    15:"No Vehicles",
    16:"Truck Prohibited",
    17:"No Entry",
    18:"Danger",
    19:"Curve Left",
    20:"Curve Right",
    21:"Double Curve",
    22:"Bumpy Road",
    23:"Slippery Road",
    24:"Road Narrows",
    25:"Road Work",
    26:"Traffic Signal",
    27:"Pedestrian",
    28:"Children Crossing",
    29:"Bicycle Crossing",
    30:"Snow",
    31:"Wild Animals",
    32:"End Restriction",
    33:"Turn Right",
    34:"Turn Left",
    35:"Ahead Only",
    36:"Straight or Right",
    37:"Straight or Left",
    38:"Keep Right",
    39:"Keep Left",
    40:"Roundabout",
    41:"End No Passing",
    42:"End No Passing Trucks"
}


@app.route("/")
def home():
    return render_template(
        "index.html",
        accuracy="93%"
    )


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    save_path = os.path.join(
        UPLOAD_FOLDER,
        "uploaded_image.png"
    )

    file.save(save_path)

    img = cv2.imread(save_path)

    img = cv2.resize(
        img,
        (48, 48)
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)

    confidence = round(
        np.max(probability) * 100,
        2
    )

    return render_template(
        "result.html",
        prediction=class_names[prediction],
        confidence=confidence,
        image_path=save_path
    )


if __name__ == "__main__":
    app.run(debug=True)
