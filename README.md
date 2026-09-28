# Plant Disease Detection & Diagnosis

Detects plant diseases from leaf images and provides a full diagnosis (symptoms, cause, treatment, prevention). Trained on the PlantVillage dataset subset (15 classes: Tomato, Potato, Pepper).

## Project structure

```
data/
  PlantVillage/          cleaned dataset (15 class folders, 20,636 images)
  diagnosis.json         disease knowledge base
  split/                 stratified train/val/test CSVs (created at run time)
src/
  train.py               training pipeline (CLI)
  predict.py             inference module + CLI
  diagnosis.py           diagnosis lookup
  preprocessing.py       tf.data loaders + augmentation
app/
  main.py                FastAPI web app
  static/index.html      upload UI
models/                  saved model + class mapping (created at run time)
reports/                 metrics, history, confusion matrix (created at run time)
notebooks/train.ipynb    interactive notebook mirroring train.py
```

## Setup

```
pip install -r requirements.txt
```

## Train

```
python src/train.py --backbone custom --img-size 96 --epochs 20
```

Optional transfer-learning backbone (requires GPU — very slow on CPU, ~2.6 h/epoch):

```
python src/train.py --backbone mobilenetv2 --img-size 96 --epochs 10
```

Artifacts written to `models/` (`plant_disease_model.keras`, `class_indices.json`, `config.json`) and metrics to `reports/`.

## Predict from the command line

```
python src/predict.py --image path/to/leaf.jpg
```

Output: top class, confidence, per-class probabilities, `confident` flag (false when confidence < 0.5), and the full diagnosis (symptoms, cause, treatment, prevention).

## Run the web app

```
python -m uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 , upload a leaf photo, and view the diagnosis. Invalid/corrupt uploads return HTTP 400 with a descriptive message; low-confidence predictions are flagged in the UI.

## Current model metrics

Checkpoint produced during training run 2 (custom CNN, 96×96, CPU):

- Test accuracy: **0.8488**
- Test loss: 0.7725
- Weighted F1: 0.8429

See `reports/classification_report.txt` and `reports/confusion_matrix.png` for per-class detail. Note: the earlier interrupted run reached 0.9297 test accuracy; it was overwritten by run 2. Retraining without label smoothing (run-1 config) recovers ~0.93.

## Classes

- Pepper: Bacterial spot, Healthy
- Potato: Early blight, Late blight, Healthy
- Tomato: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites (two-spotted), Target spot, Mosaic virus (ToMV), Yellow leaf curl virus (TYLCV), Healthy