from flask import Flask, request
import cv2
import numpy as np
import os

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image'].read()

    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([170,120,70])
    upper2 = np.array([180,255,255])

    mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.7:
            return "YES"

    return "NO"

@app.route('/')
def home():
    return "Server running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
