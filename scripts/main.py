import threading
import tkinter as tk
from tkinter import scrolledtext
from scripts.gesture_recognition import load_data, train_model
from scripts.speech_to_text import convert_speech_to_text
from utils.image_preprocessing import preprocess_image
from utils.speech_utils import recognize_speech
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load gesture recognition model
gesture_model = load_model("models/gesture_model.h5")
X, y, class_map = load_data()

class MultimodalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimodal Interaction System")
        self.root.geometry("600x400")

        # Speech Recognition Section
        self.speech_output = scrolledtext.ScrolledText(root, height=5, width=70, state="disabled")
        self.speech_output.pack(pady=10)

        self.speech_button = tk.Button(root, text="Start Speech Recognition", command=self.toggle_speech_recognition)
        self.speech_button.pack()

        # Gesture Recognition Section
        self.gesture_output = tk.Label(root, text="Gesture: None", font=("Arial", 12))
        self.gesture_output.pack(pady=10)

        self.gesture_button = tk.Button(root, text="Start Gesture Recognition", command=self.toggle_gesture_recognition)
        self.gesture_button.pack()

    def toggle_speech_recognition(self):
        threading.Thread(target=convert_speech_to_text).start()

    def toggle_gesture_recognition(self):
        threading.Thread(target=self.gesture_recognition_thread).start()

    def gesture_recognition_thread(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (64, 64))
            normalized_frame = resized_frame / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=0)

            prediction = gesture_model.predict(input_frame)
            gesture_idx = np.argmax(prediction)
            gesture_name = list(class_map.keys())[gesture_idx]

            self.gesture_output.config(text=f"Gesture: {gesture_name}")
            cv2.imshow("Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = MultimodalApp(root)
    root.mainloop()