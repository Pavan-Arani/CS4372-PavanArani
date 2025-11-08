import os
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger # type: ignore
from sklearn.model_selection import train_test_split

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default="./Assignment 3", help="Top-level dataset folder (contains 'train' and 'labels.csv')")
    p.add_argument("--output_dir", default="outputs", help="Where to save models, plots, logs")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--fine_tune_epochs", type=int, default=0, help="If >0, unfreeze base_model and continue training")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--fine_tune_lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--val_fraction", type=float, default=0.15, help="Fraction for validation (default 0.15)")
    p.add_argument("--test_fraction", type=float, default=0.15, help="Fraction for test (default 0.15)")
    p.add_argument("--use_augment", action="store_true", help="Enable simple data augmentation layers")
    return p.parse_args()

def read_kaggle_labels_csv(labels_csv_path, base_dir):
    df = pd.read_csv(labels_csv_path)
    # handle either 'id'+'breed' (Kaggle) or 'image_path'+'label'
    if 'id' in df.columns and 'breed' in df.columns:
        df['image_path'] = df['id'].astype(str) + '.jpg'
        df['label'] = df['breed']
    elif 'image_path' in df.columns and 'label' in df.columns:
        pass
    else:
        raise ValueError("CSV must contain either (id,breed) or (image_path,label) columns.")
    # full path
    df['image_path_full'] = df['image_path'].apply(lambda p: os.path.join(base_dir, 'train', p))
    # drop rows with missing files (robustness)
    df = df[df['image_path_full'].apply(os.path.exists)].reset_index(drop=True)
    return df[['image_path_full','label']]

def stratified_split(df, seed, val_frac=0.15, test_frac=0.15):
    # df columns: image_path_full, label
    if (val_frac + test_frac) >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")
    train_frac = 1.0 - (val_frac + test_frac)
    # first split off train vs temp (val+test)
    train_df, temp_df = train_test_split(df, train_size=train_frac, stratify=df['label'], random_state=seed)
    # now split temp into val and test according to relative sizes
    # temp_frac_total = val_frac + test_frac
    rel_test = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(temp_df, test_size=rel_test, stratify=temp_df['label'], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def make_tf_dataset(df, img_size, batch_size, shuffle=True, augment=False, seed=123):
    AUTOTUNE = tf.data.AUTOTUNE
    paths = df['image_path_full'].tolist()
    labels = df['label_int'].tolist()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def _load(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000, seed=seed)
    if augment:
        # simple augmentation layers applied on the fly
        def _augment(image, label):
            image = tf.image.random_flip_left_right(image, seed=seed)
            image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
            image = tf.image.random_contrast(image, 0.9, 1.1, seed=seed)
            return image, label
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def build_model(num_classes, img_size, learning_rate):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(img_size,img_size,3), include_top=False, weights='imagenet')
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(img_size,img_size,3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model, base_model

def plot_history(history, out_dir):
    plt.figure(figsize=(8,5))
    plt.plot(history.history.get('accuracy',[]), label='train_acc')
    plt.plot(history.history.get('val_accuracy',[]), label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(out_dir, 'acc_plot.png'))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(history.history.get('loss',[]), label='train_loss')
    plt.plot(history.history.get('val_loss',[]), label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(out_dir, 'loss_plot.png'))
    plt.close()

def save_25_examples_csv_and_grid(model, df_test, class_names, out_dir, img_size):
    """
    df_test: DataFrame with columns image_path_full and label_int (integers)
    Will predict on the first 25 test images (or random 25) and save a grid + CSV
    """
    n_examples = min(25, len(df_test))
    if n_examples == 0:
        print("No test examples available to save.")
        return
    sample_df = df_test.sample(n=n_examples, random_state=42).reset_index(drop=True)
    imgs = []
    trues = []
    preds = []
    confs = []
    paths = []
    for idx, row in sample_df.iterrows():
        path = row['image_path_full']
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        arr = tf.cast(img, tf.float32) / 255.0
        prob = model.predict(np.expand_dims(arr.numpy(), axis=0), verbose=0)[0]
        pred = int(np.argmax(prob))
        conf = float(np.max(prob))
        imgs.append(arr.numpy())
        trues.append(int(row['label_int']))
        preds.append(pred)
        confs.append(conf)
        paths.append(path)

    # Save CSV
    out_csv = os.path.join(out_dir, "25_examples.csv")
    recs = []
    for p,t,pr,c in zip(paths,trues,preds,confs):
        recs.append({'image_path': p, 'true_label': class_names[t], 'pred_label': class_names[pr], 'confidence': c})
    pd.DataFrame(recs).to_csv(out_csv, index=False)
    print("Saved 25 example predictions CSV to", out_csv)

    # Create grid image
    n = len(imgs)
    cols = 5; rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12,12))
    axes = axes.flatten()
    for i in range(rows*cols):
        ax = axes[i]
        if i < n:
            im = imgs[i]
            ax.imshow(np.clip(im,0,1))
            ax.axis('off')
            true_label = class_names[trues[i]] if trues[i] < len(class_names) else str(trues[i])
            pred_label = class_names[preds[i]] if preds[i] < len(class_names) else str(preds[i])
            ax.set_title(f"T:{true_label}\nP:{pred_label}\n{confs[i]:.2f}", fontsize=8)
        else:
            ax.axis('off')
    plt.tight_layout()
    out_img = os.path.join(out_dir, '25_examples.png')
    plt.savefig(out_img)
    plt.close()
    print("Saved 25 example grid to", out_img)

def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    base_dir = args.base_dir
    labels_csv = os.path.join(base_dir, 'labels.csv')
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"labels.csv not found at {labels_csv}. Ensure your dataset folder (base_dir) is correct and contains labels.csv")

    df = read_kaggle_labels_csv(labels_csv, base_dir)
    print("Total labeled images found:", len(df))

    # stratified 70/15/15 split by default (train/val/test)
    train_df, val_df, test_df = stratified_split(df, seed=args.seed, val_frac=args.val_fraction, test_frac=args.test_fraction)
    print("Split sizes: train:", len(train_df), "val:", len(val_df), "test:", len(test_df))

    # map labels to ints
    class_names = sorted(df['label'].unique().tolist())
    mapping = {lab:i for i,lab in enumerate(class_names)}
    train_df['label_int'] = train_df['label'].map(mapping)
    val_df['label_int'] = val_df['label'].map(mapping)
    test_df['label_int'] = test_df['label'].map(mapping)

    # Create tf.data datasets
    train_ds = make_tf_dataset(train_df, args.img_size, args.batch_size, shuffle=True, augment=args.use_augment, seed=args.seed)
    val_ds = make_tf_dataset(val_df, args.img_size, args.batch_size, shuffle=False, augment=False, seed=args.seed)
    test_ds = make_tf_dataset(test_df, args.img_size, args.batch_size, shuffle=False, augment=False, seed=args.seed)

    num_classes = len(class_names)
    print("Classes ({}): {}".format(num_classes, class_names[:10] + (['...'] if num_classes>10 else [])))

    model, base_model = build_model(num_classes, args.img_size, args.learning_rate)
    model.summary()

    # Callbacks
    ckpt_path = os.path.join(args.output_dir, "best_model.h5")
    checkpoint = ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    csv_logger = CSVLogger(os.path.join(args.output_dir, 'training_log.csv'))

    history = model.fit(train_ds,
                        epochs=args.epochs,
                        validation_data=val_ds,
                        callbacks=[checkpoint, reduce_lr, early, csv_logger])

    # Optionally fine-tune entire base_model
    if args.fine_tune_epochs and args.fine_tune_epochs > 0:
        base_model.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Starting fine-tuning...")
        history_fine = model.fit(train_ds,
                                 epochs=args.epochs + args.fine_tune_epochs,
                                 initial_epoch=history.epoch[-1] + 1,
                                 validation_data=val_ds,
                                 callbacks=[checkpoint, reduce_lr, early, csv_logger])
        history = history_fine

    # Save final model & plots
    final_model_path = os.path.join(args.output_dir, "final_model.h5")
    model.save(final_model_path)
    print("Saved final model to", final_model_path)
    plot_history(history, args.output_dir)

    # Evaluate on test set
    print("Evaluating on test set:")
    eval_res = model.evaluate(test_ds, verbose=1)
    print("Test evaluation (loss, accuracy):", eval_res)

    # Save 25 example grid + CSV (true label and predicted)
    save_25_examples_csv_and_grid(model, test_df, class_names, args.output_dir, args.img_size)

    # Save experiment record (append)
    exp_record = {
        "data_source": base_dir,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "epochs": args.epochs,
        "fine_tune_epochs": args.fine_tune_epochs,
        "learning_rate": args.learning_rate,
        "fine_tune_lr": args.fine_tune_lr,
        "final_test_loss": float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
        "final_test_acc": float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else None
    }
    exp_df = pd.DataFrame([exp_record])
    exp_csv = os.path.join(args.output_dir, "experiments.csv")
    if os.path.exists(exp_csv):
        old = pd.read_csv(exp_csv)
        new = pd.concat([old, exp_df], ignore_index=True)
    else:
        new = exp_df
    new.to_csv(exp_csv, index=False)
    print("Saved experiment log to", exp_csv)
    print("All outputs in:", args.output_dir)

if __name__ == "__main__":
    main()