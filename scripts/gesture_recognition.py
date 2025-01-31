import os
import cv2
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from utils.image_preprocessing import preprocess_image

def collect_gesture_data(gesture_name, save_path="datasets/gestures", num_images=200):
    """
    Collect images for a specific gesture using a webcam.
    """
    os.makedirs(f"{save_path}/{gesture_name}", exist_ok=True)
    cap = cv2.VideoCapture(0)

    print(f"Collecting images for gesture: {gesture_name}. Press 'q' to stop.")
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Collecting Gesture Data", frame)
        img_path = f"{save_path}/{gesture_name}/{gesture_name}_{count}.jpg"
        cv2.imwrite(img_path, frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {count} images for gesture: {gesture_name}")

def load_data(data_dir="datasets/gestures", img_size=(64, 64)):
    """
    Load and preprocess gesture data.
    """
    X, y = [], []
    classes = os.listdir(data_dir)
    class_map = {gesture: idx for idx, gesture in enumerate(classes)}

    for gesture, idx in class_map.items():
        gesture_path = os.path.join(data_dir, gesture)
        for img_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_name)
            img = cv2.imread(img_path)
            img = preprocess_image(img, img_size)
            X.append(img)
            y.append(idx)

    X = np.array(X)
    y = np.array(y)
    return X, y, class_map

def train_model():
    """
    Train a CNN model for gesture recognition.
    """
    X, y, class_map = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(class_map), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save("models/gesture_model.h5")
    print("Model trained and saved!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", help="Collect gesture data", action="store_true")
    parser.add_argument("--train", help="Train the gesture recognition model", action="store_true")
    args = parser.parse_args()

    if args.collect:
        gesture_name = input("Enter the gesture name: ").strip()
        collect_gesture_data(gesture_name)
    elif args.train:
        train_model()