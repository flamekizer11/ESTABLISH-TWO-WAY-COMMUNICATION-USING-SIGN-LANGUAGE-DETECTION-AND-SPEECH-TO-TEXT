# Sign Language Project

This project combines **speech-to-text** and **gesture recognition** to create a multimodal interaction system. It allows users to interact with the system using voice commands or hand gestures.

## Features
- Speech-to-text conversion using Google Speech API.
- Gesture recognition using a CNN model trained on custom datasets.
- Real-time interaction via webcam and microphone.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Collect gesture data: Run `scripts/gesture_recognition.py`.
3. Train the model: Run `scripts/gesture_recognition.py --train`.
4. Run the main application: `python scripts/main.py`.