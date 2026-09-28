import argparse
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from diagnosis import get_diagnosis

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PlantDiseasePredictor:
    def __init__(self, model_dir=None, confidence_threshold=0.5):
        model_dir = model_dir or os.path.join(BASE_DIR, "models")
        self.model = tf.keras.models.load_model(os.path.join(model_dir, "plant_disease_model.keras"))
        with open(os.path.join(model_dir, "class_indices.json"), "r") as f:
            self.class_indices = {int(k): v for k, v in json.load(f).items()}
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            self.config = json.load(f)
        self.img_size = self.config.get("img_size", 96)
        self.confidence_threshold = confidence_threshold

    def preprocess(self, image):
        if isinstance(image, str) or not hasattr(image, "convert"):
            image = Image.open(image)
        image = image.convert("RGB").resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        arr = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, image):
        x = self.preprocess(image)
        probs = self.model.predict(x, verbose=0)[0]
        order = np.argsort(probs)[::-1]
        top_idx = int(order[0])
        top_class = self.class_indices[top_idx]
        top_conf = float(probs[top_idx])
        all_probs = {self.class_indices[i]: float(probs[i]) for i in range(len(probs))}
        diagnosis = get_diagnosis(top_class, confidence=top_conf)
        return {
            "top_class": top_class,
            "top_confidence": top_conf,
            "confident": top_conf >= self.confidence_threshold,
            "probabilities": all_probs,
            "diagnosis": diagnosis,
        }


def main():
    parser = argparse.ArgumentParser(description="Predict disease on a leaf image")
    parser.add_argument("--image", required=True, help="path to leaf image")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predictor = PlantDiseasePredictor(confidence_threshold=args.threshold)
    result = predictor.predict(args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()