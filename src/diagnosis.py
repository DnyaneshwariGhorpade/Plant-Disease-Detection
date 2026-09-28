import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAGNOSIS_PATH = os.path.join(BASE_DIR, "data", "diagnosis.json")


def load_diagnosis(path=DIAGNOSIS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_diagnosis(class_name, confidence=None, path=DIAGNOSIS_PATH):
    data = load_diagnosis(path)
    entry = data.get(class_name)
    if entry is None:
        return None
    result = dict(entry)
    result["class_name"] = class_name
    if confidence is not None:
        result["confidence"] = float(confidence)
    return result