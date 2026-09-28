import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from preprocessing import DataLoader, IMG_SIZE


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def collect_data(data_dir):
    rows = []
    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(".jpg"):
                rows.append({"path": os.path.join(folder_path, fname), "label": folder})
    return pd.DataFrame(rows)


def stratified_split(df, val_frac, test_frac, seed=42):
    rest, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed
    )
    val_adjusted = val_frac / (1 - test_frac)
    train, val = train_test_split(
        rest, test_size=val_adjusted, stratify=rest["label"], random_state=seed
    )
    return train, val, test


def build_custom_cnn(img_size, num_classes):
    inp = layers.Input((img_size, img_size, 3))
    x = layers.Conv2D(32, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)


def build_mobilenetv2(img_size, num_classes, finetune=False):
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", input_shape=(img_size, img_size, 3), include_top=False
    )
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(base.input, out)
    if finetune:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
    return model


def main():
    parser = argparse.ArgumentParser(description="Train plant disease classifier")
    parser.add_argument("--data", default=os.path.join("data", "PlantVillage"))
    parser.add_argument("--out", default="models")
    parser.add_argument("--split-out", default=os.path.join("data", "split"))
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--backbone", choices=["custom", "mobilenetv2"], default="custom")
    parser.add_argument("--finetune", action="store_true", help="unfreeze top layers of backbone after head training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.split_out, exist_ok=True)
    os.makedirs(args.reports, exist_ok=True)

    df = collect_data(args.data)
    classes = sorted(df["label"].unique())
    num_classes = len(classes)
    class_indices = {i: name for i, name in enumerate(classes)}
    label_map = {name: i for i, name in enumerate(classes)}
    df["y"] = df["label"].map(label_map)
    print(f"Loaded {len(df)} images across {num_classes} classes: {classes}")

    train, val, test = stratified_split(df, val_frac=0.1, test_frac=0.1, seed=args.seed)
    print(f"Split -> train {len(train)} | val {len(val)} | test {len(test)}")
    train.to_csv(os.path.join(args.split_out, "train.csv"), index=False)
    val.to_csv(os.path.join(args.split_out, "val.csv"), index=False)
    test.to_csv(os.path.join(args.split_out, "test.csv"), index=False)
    with open(os.path.join(args.split_out, "class_indices.json"), "w") as f:
        json.dump(class_indices, f, indent=2)

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

    if args.backbone == "mobilenetv2":
        print("WARNING: MobileNetV2 on CPU is extremely slow (~2.6 h/epoch). Prefer GPU/WSL2 or backbone=custom.")
        model = build_mobilenetv2(args.img_size, num_classes)
        base_lr, head_lr = 1e-4, 1e-3
    else:
        model = build_custom_cnn(args.img_size, num_classes)
        base_lr, head_lr = 1e-3, 1e-3

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(args.out, "plant_disease_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1,
    )
    callbacks = [
        checkpoint,
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ]

    steps_train = int(np.ceil(len(train) / args.batch_size))
    steps_val = int(np.ceil(len(val) / args.batch_size))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=head_lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    print(model.summary())

    history = model.fit(
        train_ds, epochs=args.epochs, steps_per_epoch=steps_train,
        validation_data=val_ds, validation_steps=steps_val, callbacks=callbacks, verbose=1,
    )

    if args.finetune and args.backbone == "mobilenetv2":
        model = build_mobilenetv2(args.img_size, num_classes, finetune=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(
            train_ds, epochs=5, steps_per_epoch=steps_train,
            validation_data=val_ds, validation_steps=steps_val, callbacks=callbacks, verbose=1,
        )

    best = tf.keras.models.load_model(os.path.join(args.out, "plant_disease_model.keras"))
    test_loss, test_acc = best.evaluate(test_ds, verbose=1)
    print(f"TEST accuracy: {test_acc:.4f}  loss: {test_loss:.4f}")

    labels_true = test["y"].tolist()
    probs = best.predict(test_ds, verbose=1)
    preds = np.argmax(probs, axis=1)
    names = [class_indices[i] for i in range(num_classes)]
    print(classification_report(labels_true, preds, target_names=names, digits=4))

    cm = confusion_matrix(labels_true, preds)
    np.save(os.path.join(args.reports, "confusion_matrix.npy"), cm)
    with open(os.path.join(args.reports, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(labels_true, preds, target_names=names, digits=4))

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

    history_json = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    with open(os.path.join(args.reports, "history.json"), "w") as f:
        json.dump(history_json, f, indent=2)

    with open(os.path.join(args.out, "class_indices.json"), "w") as f:
        json.dump(class_indices, f, indent=2)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"img_size": args.img_size, "num_classes": num_classes, "backbone": args.backbone}, f, indent=2)

    print(f"Artifacts saved -> {os.path.abspath(args.out)}")
    print("Done.")


if __name__ == "__main__":
    main()