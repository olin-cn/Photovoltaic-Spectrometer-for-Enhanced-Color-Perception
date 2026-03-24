import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir="
    "\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/\""
)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical, register_keras_serializable
from tqdm import tqdm

try:
    from tensorflow.keras.optimizers import AdamW
    HAS_ADAMW = True
except ImportError:
    from tensorflow.keras.optimizers import Adam
    HAS_ADAMW = False

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class CONFIG:
    MODE = 3
    IMG_SIZE = 32
    PATCH_SIZE = 4
    N_SPECTRAL_BANDS = 24
    SPECTRAL_WIDTH = 50
    PROJECTION_DIM = 192
    TRANSFORMER_LAYERS = 6
    NUM_HEADS = 8
    MLP_RATIO = 4
    EPOCHS = 1
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    DROPOUT_RATE = 0.3
    WEIGHT_DECAY = 1e-3
    SAVE_DIR = "results"
    CACHE_DIR = "spectral_cache"
    LABEL_SMOOTHING = 0.2

def add_noise(image, noise_factor=0.1):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)

def apply_augmentation(images):
    augmented = []
    for img in images:
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
        augmented.append(img)
    return np.array(augmented, dtype=np.float32)

def apply_protanopia(rgb_imgs):
    matrix = np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998]
    ], dtype=np.float32)

    transformed = np.dot(rgb_imgs, matrix.T) * 0.7 + 0.2
    transformed = np.clip(
        transformed + np.random.normal(0, 0.15, transformed.shape),
        0, 1
    )
    transformed = transformed * 0.8 + 0.1
    return transformed.astype(np.float32)

def apply_spectral_conversion(images, tag):
    os.makedirs(CONFIG.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CONFIG.CACHE_DIR, f"spectral_{tag}.npy")

    if os.path.exists(cache_path):
        print(f"Loading cache: {cache_path}")
        spectral_data = np.load(cache_path).astype(np.float32)
    else:
        print(f"Generating new spectral data: {tag}")
        images = np.array([add_noise(img, 0.03) for img in images], dtype=np.float32)
        hs_images = np.array(
            [rgb_to_highres_spectral(img) for img in tqdm(images, desc=f"{tag} spectral conversion")],
            dtype=np.float32
        )

        flattened = hs_images.reshape(-1, CONFIG.N_SPECTRAL_BANDS)
        processor = PCA(n_components=12)
        enhanced = processor.fit_transform(flattened)

        scaler = StandardScaler()
        enhanced = scaler.fit_transform(enhanced)

        spectral_data = enhanced.reshape(
            len(images),
            CONFIG.IMG_SIZE,
            CONFIG.IMG_SIZE,
            -1
        ).astype(np.float32)

        np.save(cache_path, spectral_data)

    return spectral_data.astype(np.float32)

def rgb_to_highres_spectral(rgb_img):
    wavelengths = np.linspace(400, 1000, CONFIG.N_SPECTRAL_BANDS)
    curves = [
        np.exp(-(wavelengths - 450) ** 2 / (2 * 30 ** 2)),
        np.exp(-(wavelengths - 550) ** 2 / (2 * 40 ** 2)),
        np.exp(-(wavelengths - 650) ** 2 / (2 * 35 ** 2)),
        0.7 * np.exp(-(wavelengths - 850) ** 2 / (2 * 50 ** 2))
    ]
    spectral_matrix = np.array(curves, dtype=np.float32).T

    rgb_ext = np.concatenate([
        rgb_img,
        np.mean(rgb_img, axis=-1, keepdims=True)
    ], axis=-1).astype(np.float32)

    if rgb_ext.shape[-1] != spectral_matrix.shape[1]:
        raise ValueError(
            f"Input channel mismatch: rgb_ext.shape[-1]={rgb_ext.shape[-1]}, "
            f"spectral_matrix.shape[1]={spectral_matrix.shape[1]}"
        )

    hs = np.einsum('...c,bc->...b', rgb_ext, spectral_matrix).astype(np.float32)

    photon_noise = np.sqrt(np.abs(hs)) * np.random.normal(0, 0.005, hs.shape)
    read_noise = np.random.normal(0, 0.002, hs.shape)
    hs = hs + photon_noise + read_noise

    for idx in [0, 5, 12, 18]:
        hs[..., idx] *= 1.3 + np.random.normal(0, 0.1)

    return hs.astype(np.float32)

@register_keras_serializable()
class PositionEmbedding(layers.Layer):
    def __init__(self, max_length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.embedding = layers.Embedding(input_dim=max_length, output_dim=embedding_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        pos_embed = self.embedding(positions)
        pos_embed = tf.expand_dims(pos_embed, axis=0)
        return pos_embed

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim
        })
        return config

def create_optimizer():
    if HAS_ADAMW:
        try:
            return AdamW(
                learning_rate=CONFIG.LEARNING_RATE,
                weight_decay=CONFIG.WEIGHT_DECAY
            )
        except TypeError:
            return AdamW(
                learning_rate=CONFIG.LEARNING_RATE
            )
    else:
        return Adam(
            learning_rate=CONFIG.LEARNING_RATE
        )

def create_enhanced_vit():
    input_channels = 12 if CONFIG.MODE == 2 else 3
    inputs = layers.Input(shape=(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, input_channels))

    x = layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="swish",
        kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
    )(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        CONFIG.PROJECTION_DIM,
        kernel_size=CONFIG.PATCH_SIZE,
        strides=CONFIG.PATCH_SIZE,
        padding="same",
        kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
    )(x)
    x = layers.Reshape((-1, CONFIG.PROJECTION_DIM))(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    pos_embed = PositionEmbedding(
        max_length=1000,
        embedding_dim=CONFIG.PROJECTION_DIM
    )(x)
    x = x + pos_embed

    for _ in range(CONFIG.TRANSFORMER_LAYERS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn = layers.MultiHeadAttention(
            num_heads=CONFIG.NUM_HEADS,
            key_dim=CONFIG.PROJECTION_DIM // CONFIG.NUM_HEADS,
            dropout=CONFIG.DROPOUT_RATE,
            kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
        )(x1, x1)
        x = layers.Add()([x, attn])

        x2 = layers.LayerNormalization(epsilon=1e-6)(x)
        x2 = layers.Dense(
            CONFIG.PROJECTION_DIM * CONFIG.MLP_RATIO,
            activation="swish",
            kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
        )(x2)
        x2 = layers.Dropout(CONFIG.DROPOUT_RATE)(x2)
        x2 = layers.Dense(
            CONFIG.PROJECTION_DIM,
            kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
        )(x2)
        x = layers.Add()([x, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(
        256,
        activation="swish",
        kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
    )(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(
        10,
        activation="softmax",
        kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
    )(x)

    optimizer = create_optimizer()

    loss_fn = CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=CONFIG.LABEL_SMOOTHING
    )

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    os.makedirs(CONFIG.SAVE_DIR, exist_ok=True)

    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = (x_train / 255.0).astype(np.float32)
    x_test = (x_test / 255.0).astype(np.float32)
    y_train, y_test = y_train.flatten(), y_test.flatten()

    if CONFIG.MODE == 2:
        x_train = apply_spectral_conversion(apply_augmentation(x_train), tag="train")
        x_test = apply_spectral_conversion(x_test, tag="test")
    elif CONFIG.MODE == 3:
        x_train = apply_protanopia(apply_augmentation(x_train))
        x_test = apply_protanopia(x_test)
    else:
        x_train = apply_augmentation(x_train)

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(CONFIG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = val_ds.batch(CONFIG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = create_enhanced_vit()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CONFIG.SAVE_DIR, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    model = tf.keras.models.load_model(
        os.path.join(CONFIG.SAVE_DIR, "best_model.keras"),
        custom_objects={"PositionEmbedding": PositionEmbedding}
    )

    test_loss, test_acc = model.evaluate(val_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    model.save(os.path.join(CONFIG.SAVE_DIR, "model.keras"))

    epochs_ran = len(history.history["accuracy"])
    df = pd.DataFrame({
        "epoch": list(range(1, epochs_ran + 1)),
        "train_accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"]
    })
    df.to_csv(os.path.join(CONFIG.SAVE_DIR, "history.csv"), index=False)

    y_train_true = np.argmax(y_train, axis=1)
    y_test_true = np.argmax(y_test, axis=1)

    y_train_pred = np.argmax(model.predict(train_ds, verbose=1), axis=1)
    y_test_pred = np.argmax(model.predict(val_ds, verbose=1), axis=1)

    cm_train = confusion_matrix(y_train_true, y_train_pred)
    cm_test = confusion_matrix(y_test_true, y_test_pred)

    cm_train_percentage = cm_train.astype("float32") / cm_train.sum(axis=1, keepdims=True)
    cm_test_percentage = cm_test.astype("float32") / cm_test.sum(axis=1, keepdims=True)

    pd.DataFrame(cm_train_percentage).to_csv(
        os.path.join(CONFIG.SAVE_DIR, "confusion_matrix_train_percentage.csv"),
        index=False
    )
    pd.DataFrame(cm_test_percentage).to_csv(
        os.path.join(CONFIG.SAVE_DIR, "confusion_matrix_test_percentage.csv"),
        index=False
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_accuracy"], label="Train Acc")
    plt.plot(df["epoch"], df["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.savefig(os.path.join(CONFIG.SAVE_DIR, "training_curves.png"))
    plt.close()

    if HAS_ADAMW:
        print("Optimizer: AdamW")
    else:
        print("Current TensorFlow does not support AdamW, fallback to Adam")
