import imgsim
import cv2
import numpy as np
import os
from itertools import combinations

def load_image(image_path: str) -> np.ndarray:
    return cv2.imread(image_path)
     
def vectorize_image(image: np.ndarray, vtr=imgsim.Vectorizer()) -> np.ndarray:
    return vtr.vectorize(image)

def load_and_vectorize_image(image_path: str, vtr=imgsim.Vectorizer()) -> np.ndarray:
    return vectorize_image(load_image(image_path), vtr)

def image_distance(image_a_vec: np.ndarray, image_b_vec: np.ndarray) -> float:
    return imgsim.distance(image_a_vec, image_b_vec)
