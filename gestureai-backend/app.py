# app.py
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
import numpy as np
from loader import load_model
import os
import tempfile
import cv2
import time

app = Flask(__name__)
model = load_model("./v13/checkpoint.pth.tar")
gesture_labels = ['Swipe Left', 'Swipe Right', 'Zoom In', 'Zoom Out', 'Push', 'Pull', 'No Gesture']

def preprocess_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112)) 
        frame = frame / 255.0
        frames.append(frame)
    cap.release()

    if len(frames) < 16:
        return None 

    clip = np.stack(frames[:16], axis=0)  # (T, H, W, C)
    clip = clip.transpose(3, 0, 1, 2)  # (C, T, H, W)
    tensor = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)  # (1, C, T, H, W)
    return tensor

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video = request.files['video']
    print("Video received")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video.save(tmp.name)
        input_tensor = preprocess_video(tmp.name)
        os.remove(tmp.name)

    if input_tensor is None:
        return jsonify({'error': 'Video too short or corrupt'}), 400

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        label = gesture_labels[pred]

    return jsonify({'gesture': label})


@app.route('/predict_camera', methods=['GET'])
def predict_camera():
    cap = cv2.VideoCapture(0)
    frames = []

    for _ in range(16):  # capture 16 frames for one gesture clip
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)
        time.sleep(0.5)  # give camera time between frames

    cap.release()
    print(frames)
    if len(frames) < 16:
        return jsonify({'error': 'Not enough frames'}), 400

    clip = np.stack(frames, axis=0)  # (T, H, W, C)
    clip = clip / 255.0
    clip = clip.transpose(3, 0, 1, 2)  # (C, T, H, W)
    input_tensor = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        label = gesture_labels[pred]

    return jsonify({'gesture': label})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)



































# from flask import Flask, request, jsonify
# import numpy as np
# import cv2
# import mediapipe as mp
# import base64

# app = Flask(__name__)
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_drawing = mp.solutions.drawing_utils

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.json
#         img_data = base64.b64decode(data["image"])
#         np_arr = np.frombuffer(img_data, np.uint8)
#         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         result = hands.process(img_rgb)

#         if result.multi_hand_landmarks:
#             return jsonify({"gesture": "Hand Detected", "confidence": 0.95})
#         else:
#             return jsonify({"gesture": "No Hand", "confidence": 0.0})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
