import cv2
import numpy as np

def preprocess_image(image, target_size=(64, 64)):
    """
    Preprocess an image for gesture recognition.
    """
    resized_image = cv2.resize(image, target_size)
    normalized_image = resized_image / 255.0
    return normalized_image