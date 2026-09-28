import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import DataLoader


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Resume training from an existing checkpoint")
    parser.add_argument("--model", default=os.path.join("models", "plant_disease_model.keras"))
    parser.add_argument("--split-dir", default=os.path.join("data", "split"))
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.reports, exist_ok=True)

    with open(os.path.join(args.split_dir, "class_indices.json")) as f:
        class_indices = {int(k): v for k, v in json.load(f).items()}
    num_classes = len(class_indices)
    names = [class_indices[i] for i in range(num_classes)]

    train = pd.read_csv(os.path.join(args.split_dir, "train.csv"))
    val = pd.read_csv(os.path.join(args.split_dir, "val.csv"))
    test = pd.read_csv(os.path.join(args.split_dir, "test.csv"))

    train_ds = DataLoader(
        train["path"].tolist(), train["y"].tolist(), num_classes,
        img_size=args.img_size, batch_size=args.batch_size, augment=True,
    ).get_dataset(repeat=True)
    val_ds = DataLoader(
        val["path"].tolist(), val["y"].tolist(), num_classes,
        img_size=args.img_size, batch_size=args.batch_size,
    ).get_dataset()
    test_ds = DataLoader(
        test["path"].tolist(), test["y"].tolist(), num_classes,
        img_size=args.img_size, batch_size=args.batch_size,
    ).get_dataset()

    print(f"Loading checkpoint: {args.model}")
    model = tf.keras.models.load_model(args.model)
    assert model.output_shape[-1] == num_classes, "class mismatch between split and model"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.model, monitor="val_accuracy", save_best_only=True, verbose=1,
    )
    callbacks = [
        checkpoint,
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ]

    steps_train = int(np.ceil(len(train) / args.batch_size))
    steps_val = int(np.ceil(len(val) / args.batch_size))

    print(f"Resuming training (no label smoothing) -> train {len(train)} | val {len(val)} | test {len(test)}")
    history = model.fit(
        train_ds, epochs=args.epochs, steps_per_epoch=steps_train,
        validation_data=val_ds, validation_steps=steps_val, callbacks=callbacks, verbose=1,
    )

    best = tf.keras.models.load_model(args.model)
    test_loss, test_acc = best.evaluate(test_ds, verbose=1)
    print(f"TEST accuracy: {test_acc:.4f}  loss: {test_loss:.4f}")

    labels_true = test["y"].tolist()
    probs = best.predict(test_ds, verbose=0)
    preds = np.argmax(probs, axis=1)
    with open(os.path.join(args.reports, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("Resumed training (no label smoothing)\n\n")
        f.write(classification_report(labels_true, preds, target_names=names, digits=4))

    cm = confusion_matrix(labels_true, preds)
    np.save(os.path.join(args.reports, "confusion_matrix.npy"), cm)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(names, rotation=90, fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.reports, "confusion_matrix.png"), dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"Could not render confusion matrix plot: {exc}")

    prev = {}
    hist_path = os.path.join(args.reports, "history.json")
    if os.path.exists(hist_path):
        try:
            with open(hist_path) as f:
                prev = json.load(f)
        except Exception:
            prev = {}
    merged = {k: prev.get(k, []) + [float(v) for v in vs] for k, vs in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Best checkpoint saved -> {os.path.abspath(args.model)}")
    print("Done.")


if __name__ == "__main__":
    main()