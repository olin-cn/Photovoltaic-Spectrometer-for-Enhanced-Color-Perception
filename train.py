import os
import time
import json

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
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical, register_keras_serializable
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class CONFIG:
    MODE = # 1: RGB, 2: spectral, 3: protanopia
    IMG_SIZE = 32
    PATCH_SIZE = 4
    N_SPECTRAL_BANDS = 24
    PROJECTION_DIM = 192
    TRANSFORMER_LAYERS = 6
    NUM_HEADS = 8
    MLP_RATIO = 4
    EPOCHS = 1
    BATCH_SIZE = 128
    LEARNING_RATE = 
    DROPOUT_RATE = 
    WEIGHT_DECAY = 
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
        if CONFIG.MODE != 2 and np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            img = tf.keras.preprocessing.image.apply_affine_transform(
                img, theta=angle, row_axis=0, col_axis=1, channel_axis=2
            )
        augmented.append(img)
    return np.array(augmented, dtype=np.float32)

def apply_protanopia(rgb_imgs):
    matrix = np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998]
    ], dtype=np.float32)
    transformed = np.dot(rgb_imgs, matrix.T) * 0.7 + 0.2
    transformed = np.clip(transformed + np.random.normal(0, 0.15, transformed.shape), 0, 1)
    transformed = transformed * 0.8 + 0.1
    return transformed.astype(np.float32)

def rgb_to_highres_spectral(rgb_img):
    wavelengths = np.linspace(400, 1000, CONFIG.N_SPECTRAL_BANDS).astype(np.float32)
    curves = [
        np.exp(-(wavelengths - 450) ** 2 / (2 * 30 ** 2)),
        np.exp(-(wavelengths - 550) ** 2 / (2 * 40 ** 2)),
        np.exp(-(wavelengths - 650) ** 2 / (2 * 35 ** 2)),
        0.7 * np.exp(-(wavelengths - 850) ** 2 / (2 * 50 ** 2))
    ]
    spectral_matrix = np.array(curves, dtype=np.float32).T
    rgb_ext = np.concatenate([
        rgb_img.astype(np.float32),
        np.mean(rgb_img, axis=-1, keepdims=True).astype(np.float32)
    ], axis=-1)
    if rgb_ext.shape[-1] != spectral_matrix.shape[1]:
        raise ValueError(
            f"channel mismatch: rgb_ext={rgb_ext.shape[-1]}, spectral_matrix={spectral_matrix.shape[1]}"
        )
    hs = np.einsum('...c,bc->...b', rgb_ext, spectral_matrix).astype(np.float32)
    photon_noise = (np.sqrt(np.abs(hs)) * np.random.normal(0, 0.005, hs.shape)).astype(np.float32)
    read_noise = np.random.normal(0, 0.002, hs.shape).astype(np.float32)
    hs = hs + photon_noise + read_noise
    for idx in [0, 5, 12, 18]:
        hs[..., idx] *= (1.3 + np.random.normal(0, 0.1)).astype(np.float32)
    return hs.astype(np.float32)

def apply_spectral_conversion(images, tag):
    os.makedirs(CONFIG.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CONFIG.CACHE_DIR, f"spectral_{tag}.npy")
    if os.path.exists(cache_path):
        print(f"loading cache: {cache_path}")
        spectral_data = np.load(cache_path)
        return spectral_data.astype(np.float32)

    print(f"creating spectral data: {tag}")
    images = np.array([add_noise(img, 0.03) for img in images], dtype=np.float32)
    hs_images = np.array(
        [rgb_to_highres_spectral(img) for img in tqdm(images, desc=f'{tag}_spectral')],
        dtype=np.float32
    )

    flattened = hs_images.reshape(-1, CONFIG.N_SPECTRAL_BANDS).astype(np.float32)
    processor = PCA(n_components=12)
    enhanced = processor.fit_transform(flattened).astype(np.float32)

    scaler = StandardScaler()
    enhanced = scaler.fit_transform(enhanced).astype(np.float32)

    spectral_data = enhanced.reshape(len(images), CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, -1).astype(np.float32)
    np.save(cache_path, spectral_data.astype(np.float16))
    return spectral_data

@register_keras_serializable()
class SpectralAttention(layers.Layer):
    def __init__(self, num_heads=8, key_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
        )

    def call(self, inputs):
        attended = self.attn(inputs, inputs)
        return inputs + attended

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return config

@register_keras_serializable()
class PositionEmbedding(layers.Layer):
    def __init__(self, max_length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.embedding = layers.Embedding(input_dim=max_length, output_dim=embedding_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        return self.embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"max_length": self.max_length, "embedding_dim": self.embedding_dim})
        return config

def get_optimizer():
    try:
        from tensorflow.keras.optimizers import AdamW
        return AdamW(learning_rate=CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY)
    except Exception:
        from tensorflow.keras.optimizers.experimental import AdamW
        return AdamW(learning_rate=CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY)

def create_model(mode):
    input_channels = 12 if mode == 2 else 3
    inputs = layers.Input(shape=(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, input_channels))

    if mode == 2:
        x = layers.Conv2D(64, kernel_size=3, padding='same', activation='swish',
                          kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY))(inputs)
        x = layers.BatchNormalization()(x)
    elif mode == 1:
        x = layers.Conv2D(64, kernel_size=3, padding='same', activation='swish',
                          kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY))(inputs)
    else:
        x = layers.Conv2D(48, kernel_size=3, padding='same', activation='swish',
                          kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY))(inputs)

    x = layers.Conv2D(CONFIG.PROJECTION_DIM, kernel_size=CONFIG.PATCH_SIZE,
                      strides=CONFIG.PATCH_SIZE, padding='same',
                      kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY))(x)
    x = layers.Reshape((-1, CONFIG.PROJECTION_DIM))(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    if mode == 2:
        x = SpectralAttention()(x)

    pos_embed = PositionEmbedding(max_length=1000, embedding_dim=CONFIG.PROJECTION_DIM)(x)
    x = x + pos_embed

    for _ in range(CONFIG.TRANSFORMER_LAYERS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn = layers.MultiHeadAttention(
            num_heads=CONFIG.NUM_HEADS,
            key_dim=CONFIG.PROJECTION_DIM // CONFIG.NUM_HEADS,
            dropout=CONFIG.DROPOUT_RATE,
            kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
        )(x1, x1)
        x = x + attn

        x2 = layers.LayerNormalization(epsilon=1e-6)(x)
        x2 = layers.Dense(CONFIG.PROJECTION_DIM * CONFIG.MLP_RATIO, activation='swish',
                          kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY))(x2)
        x2 = layers.Dropout(CONFIG.DROPOUT_RATE)(x2)
        x2 = layers.Dense(CONFIG.PROJECTION_DIM,
                          kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY))(x2)
        x = x + x2

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    if mode == 2:
        x = layers.Dense(384, activation='swish',
                         kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY))(x)
        x = layers.BatchNormalization()(x)
    elif mode == 1:
        x = layers.Dense(256, activation='swish',
                         kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY))(x)
    else:
        x = layers.Dense(192, activation='swish',
                         kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY))(x)

    outputs = layers.Dense(10, activation='softmax',
                           kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY))(x)

    model = models.Model(inputs, outputs)
    loss_fn = CategoricalCrossentropy(from_logits=False, label_smoothing=CONFIG.LABEL_SMOOTHING)
    model.compile(optimizer=get_optimizer(), loss=loss_fn, metrics=['accuracy'])
    return model

def try_get_flops(model, input_shape):
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
        f = tf.function(model.call)
        concrete = f.get_concrete_function(tf.TensorSpec([1] + list(input_shape), model.inputs[0].dtype))
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            return int(flops.total_float_ops) if flops is not None else None
    except Exception:
        return None

def make_datasets(x_train, y_train, x_test, y_test):
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(CONFIG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(CONFIG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_eval_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(CONFIG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, train_eval_ds

def compute_confusion_matrix(model, ds):
    y_true = []
    y_pred = []
    for xb, yb in ds:
        pb = model.predict(xb, verbose=0)
        y_true.append(np.argmax(yb.numpy(), axis=1))
        y_pred.append(np.argmax(pb, axis=1))
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float32') / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    return cm, cm_pct

if __name__ == '__main__':
    os.makedirs(CONFIG.SAVE_DIR, exist_ok=True)

    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train / 255.0).astype(np.float32)
    x_test = (x_test / 255.0).astype(np.float32)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    if CONFIG.MODE == 2:
        x_train = apply_spectral_conversion(apply_augmentation(x_train), tag='train')
        x_test = apply_spectral_conversion(x_test, tag='test')
    elif CONFIG.MODE == 3:
        x_train = apply_protanopia(apply_augmentation(x_train))
        x_test = apply_protanopia(x_test)
    else:
        x_train = apply_augmentation(x_train)

    y_train_oh = to_categorical(y_train, num_classes=10)
    y_test_oh = to_categorical(y_test, num_classes=10)

    train_ds, val_ds, train_eval_ds = make_datasets(x_train, y_train_oh, x_test, y_test_oh)

    model = create_model(CONFIG.MODE)
    model.summary()

    input_shape = (CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, 12 if CONFIG.MODE == 2 else 3)
    flops = try_get_flops(model, input_shape)
    params = int(model.count_params())

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CONFIG.SAVE_DIR, f'best_model_mode{CONFIG.MODE}.keras'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

    t0 = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    train_seconds = time.perf_counter() - t0

    best_path = os.path.join(CONFIG.SAVE_DIR, f'best_model_mode{CONFIG.MODE}.keras')
    model = tf.keras.models.load_model(best_path)

    t1 = time.perf_counter()
    test_loss, test_acc = model.evaluate(val_ds, verbose=0)
    eval_seconds = time.perf_counter() - t1
    print(f"test_acc: {test_acc:.4f}")

    model.save(os.path.join(CONFIG.SAVE_DIR, f'model_mode{CONFIG.MODE}.keras'))

    epochs_ran = len(history.history['accuracy'])
    df = pd.DataFrame({
        'epoch': list(range(1, epochs_ran + 1)),
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    })
    df.to_csv(os.path.join(CONFIG.SAVE_DIR, f'history_mode{CONFIG.MODE}.csv'), index=False)

    cm_train, cm_train_pct = compute_confusion_matrix(model, train_eval_ds)
    cm_test, cm_test_pct = compute_confusion_matrix(model, val_ds)

    pd.DataFrame(cm_train_pct).to_csv(os.path.join(CONFIG.SAVE_DIR, f'cm_train_pct_mode{CONFIG.MODE}.csv'), index=False)
    pd.DataFrame(cm_test_pct).to_csv(os.path.join(CONFIG.SAVE_DIR, f'cm_test_pct_mode{CONFIG.MODE}.csv'), index=False)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_accuracy'], label='Train Acc')
    plt.plot(df['epoch'], df['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(CONFIG.SAVE_DIR, f'training_curves_mode{CONFIG.MODE}.png'))
    plt.close()

    summary = {
        "mode": int(CONFIG.MODE),
        "params": params,
        "flops_batch1": flops,
        "train_seconds": float(train_seconds),
        "eval_seconds": float(eval_seconds),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "config": {k: getattr(CONFIG, k) for k in dir(CONFIG) if k.isupper()}
    }
    with open(os.path.join(CONFIG.SAVE_DIR, f"summary_mode{CONFIG.MODE}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
