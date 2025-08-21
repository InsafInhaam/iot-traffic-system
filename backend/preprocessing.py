import cv2
import numpy as np

def resize_image(image: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """Resize an image to the given width and height."""
    return cv2.resize(image, (width, height))

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def blur_image(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def edge_detection(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """Detect edges using Canny edge detection."""
    return cv2.Canny(image, low_threshold, high_threshold)

def preprocess_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Example pipeline:
    1. Resize
    2. Grayscale
    3. Blur
    4. Edge detection
    """
    resized = resize_image(image)
    gray = to_grayscale(resized)
    blurred = blur_image(gray)
    edges = edge_detection(blurred)
    return edges
