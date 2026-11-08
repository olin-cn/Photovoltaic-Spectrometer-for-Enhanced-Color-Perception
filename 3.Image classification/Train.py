
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

try:
    import tensorflow as tf
except ImportError as exc:  # keep the error readable before the rest imports
    raise RuntimeError("Install TensorFlow before running this file.") from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
)


class CONFIG:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data" / "Houston2018"
    SAVE_DIR = BASE_DIR / "results_houston_tensorflow"
    IMG_SIZE = 32
    TILE_SIZE = 64
    LOCAL_PATCH_SIZE = 31
    SOURCE_CLASSES = 16
    CLASSES = 10
    TRAIN_PER_CLASS = 1000
    VAL_PER_CLASS = 250
    TEST_MAX = 40000
    TEST_MIN_PER_CLASS = 10
    SEED = 42
    EPOCHS = 50
    BATCH_SIZE = 32
    PEAK_LEARNING_RATE = 5e-5
    MIN_LEARNING_RATE = 1e-6
    LR_WARMUP_EPOCHS = 3
    WEIGHT_DECAY = 1e-2
    LABEL_SMOOTHING = .05
    DROPOUT = .20
    DROP_PATH = .10
    EMBED_DIM = 96
    TRANSFORMER_LAYERS = 3
    NUM_HEADS = 4
    MLP_RATIO = 3
    STEM_CHANNELS = 32
    MIXUP_PROB = .20
    MIXUP_ALPHA = .20
    EMA_DECAY = .99
    AUG_NOISE_STD = .01
    AUG_NOISE_PROB = .50
    REQUIRE_ALL_CLASSES = True
    MIN_REGION_CLASS_COUNT = 10
    SPLIT_SOLVER_SECONDS = 60
    WAVELENGTH_FILE = None
    ASSUMED_HSI_WL_MIN = 380.0
    ASSUMED_HSI_WL_MAX = 1000.0


PROFILE = "centered_small"
MODE_NAMES = {1: "mode_1_rgb", 2: "mode_2_native", 3: "mode_3_protanopia"}
MODE_CHANNELS = {1: 3, 2: 12, 3: 3}
MODE2_BAND_INDICES = np.asarray(
    [0, 4, 9, 13, 18, 22, 27, 31, 36, 40, 45, 49], dtype=np.int32
)
EXPECTED_SPLIT_SHA256 = "e91e11f08255d957b57ff92a91ae2b3fea60c9ced2bf7b9e40d9173f96bbe46f"
EXPECTED_GT_SHA256 = "3d287d75afc131717b634179ae4476e7d0c8cd3e1149ee8282a80a03f4ce5910"
EXPECTED_HSI_SHA256 = "84588a62ca5feb936ed45e17566be9721ce2620bef3585ec299bb4b4aa5dcb4d"
CLASS_IDS = [1, 2, 3, 4, 8, 9, 11, 14, 15, 16]
EXCLUDED_CLASS_IDS = sorted(set(range(1, CONFIG.SOURCE_CLASSES + 1)) - set(CLASS_IDS))
CLASS_TO_INDEX = {class_id: index for index, class_id in enumerate(CLASS_IDS)}
CLASS_SELECTION_BASIS = (
    "Original subset: previous Mode2 validation recall; replaced class 7 with "
    "class 11 after inspecting original test results. Replacement chosen by "
    "train/validation spatial tile coverage, not candidate test accuracy."
)
CLASS_REVISION = {
    "removed_class_id": 7,
    "added_class_id": 11,
    "candidate_class_ids": [5, 6, 10, 11, 12, 13],
    "selection_rule": "maximize minimum of train and validation occupied tile counts",
    "added_class_train_samples": 1000,
    "added_class_validation_samples": 250,
    "added_class_train_tiles": 133,
    "added_class_validation_tiles": 102,
    "test_inspected_before_class_revision": True,
    "evaluation_status": "exploratory on existing test split",
}
LOGGER = logging.getLogger("houston_tensorflow")


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8")


def sha_json(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True).encode()).hexdigest()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_logger(directory):
    directory.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(directory / "run.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        LOGGER.addHandler(handler)


def load_gt(data_dir):
    raw = sio.loadmat(data_dir / "houston_gt.mat", variable_names=["houston_gt"])["houston_gt"]
    gt = np.where((raw >= 1) & (raw <= CONFIG.SOURCE_CLASSES), raw, 0).astype(np.uint8)
    digest = hashlib.sha256(np.ascontiguousarray(gt).tobytes()).hexdigest()
    if digest != EXPECTED_GT_SHA256:
        raise ValueError("Ground truth fingerprint differs from the locked Houston2018 data")
    return gt


def load_hsi(data_dir):
    hsi = sio.loadmat(data_dir / "houston_hsi.mat", variable_names=["houston_hsi"])["houston_hsi"]
    if hsi.ndim != 3 or hsi.shape[-1] != 50:
        raise ValueError("Expected Houston HSI with shape [height, width, 50]")
    digest = hashlib.sha256(np.ascontiguousarray(hsi).tobytes()).hexdigest()
    if digest != EXPECTED_HSI_SHA256:
        raise ValueError("Raw HSI fingerprint differs from the locked Houston2018 data")
    return hsi


def split_digest(train, val, test):
    digest = hashlib.sha256()
    for coords in (train, val, test):
        digest.update(np.ascontiguousarray(coords).tobytes())
    return digest.hexdigest()


def verify_split(train, val, test, gt):
    h, w = gt.shape
    ncols = math.ceil(w / CONFIG.TILE_SIZE)
    sets = []
    for name, coords in (("train", train), ("val", val), ("test", test)):
        if coords.ndim != 2 or coords.shape[1] != 2 or not len(coords):
            raise ValueError(f"Invalid {name} coordinates")
        r, c = coords[:, 0], coords[:, 1]
        if (r.min() < 0 or c.min() < 0 or r.max() >= h or c.max() >= w or
                not np.all(gt[r, c] > 0) or len(np.unique(coords, axis=0)) != len(coords)):
            raise ValueError(f"Invalid {name} coordinate values")
        tiles = (r // CONFIG.TILE_SIZE) * ncols + c // CONFIG.TILE_SIZE
        sets.append(set(tiles.tolist()))
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Train/validation/test spatial tiles overlap")


def _tile_histograms(gt):
    h, w = gt.shape
    before = CONFIG.IMG_SIZE // 2
    after = CONFIG.IMG_SIZE - CONFIG.IMG_SIZE // 2 - 1
    coords = np.argwhere(gt > 0).astype(np.int32)
    r, c = coords[:, 0], coords[:, 1]
    eligible = ((r >= before) & (c >= before) & (r + after < h) & (c + after < w) &
                (r % CONFIG.TILE_SIZE >= before) & (c % CONFIG.TILE_SIZE >= before) &
                (r % CONFIG.TILE_SIZE + after < CONFIG.TILE_SIZE) &
                (c % CONFIG.TILE_SIZE + after < CONFIG.TILE_SIZE))
    coords = coords[eligible]
    ncols = math.ceil(w / CONFIG.TILE_SIZE)
    tile = (coords[:, 0] // CONFIG.TILE_SIZE) * ncols + coords[:, 1] // CONFIG.TILE_SIZE
    hist = np.zeros((math.ceil(h / CONFIG.TILE_SIZE) * ncols, CONFIG.SOURCE_CLASSES), dtype=np.int64)
    labels = gt[coords[:, 0], coords[:, 1]].astype(int) - 1
    np.add.at(hist, (tile, labels), 1)
    active = np.flatnonzero(hist.sum(1) > 0)
    return coords, tile, hist, active


def _partition_tiles(hist, active):
    counts = (hist[active] > 0).sum(0)
    if np.any(counts < 3):
        raise ValueError("The locked three-way split is infeasible for a rare class")
    nb = len(active)
    rows, cols, vals, lower, upper = [], [], [], [], []
    for b in range(nb):
        row = len(lower)
        for s in range(3):
            rows.append(row); cols.append(3 * b + s); vals.append(1.)
        lower.append(1.); upper.append(1.)
    bc = hist[active].astype(float)
    for s in range(3):
        for c in range(CONFIG.SOURCE_CLASSES):
            row = len(lower)
            for b in np.flatnonzero(bc[:, c]):
                rows.append(row); cols.append(3 * int(b) + s); vals.append(float(bc[b, c]))
            lower.append(float(CONFIG.MIN_REGION_CLASS_COUNT)); upper.append(np.inf)
    for s, (lo, hi) in enumerate(((.35, .8), (.08, .35), (.08, .35))):
        row = len(lower)
        for b in range(nb):
            rows.append(row); cols.append(3 * b + s); vals.append(1.)
        lower.append(max(1, math.floor(lo * nb))); upper.append(max(1, math.ceil(hi * nb)))
    matrix = coo_matrix((vals, (rows, cols)), shape=(len(lower), nb * 3)).tocsr()
    rng = np.random.default_rng(CONFIG.SEED)
    result = milp(c=rng.uniform(.1, 1., nb * 3), integrality=np.ones(nb * 3), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
                  options={"time_limit": CONFIG.SPLIT_SOLVER_SECONDS, "mip_rel_gap": .05})
    if result.x is None:
        raise RuntimeError(f"Spatial split MILP failed: {result.message}")
    return np.argmax(np.rint(result.x).reshape(nb, 3), axis=1)


def _cap_per_class(coords, gt, limit, seed):
    rng = np.random.default_rng(seed)
    labels = gt[coords[:, 0], coords[:, 1]]
    chosen = []
    for c in range(1, CONFIG.SOURCE_CLASSES + 1):
        ids = np.flatnonzero(labels == c)
        if len(ids):
            chosen.append(rng.choice(ids, min(limit, len(ids)), replace=False))
    ids = np.concatenate(chosen)
    rng.shuffle(ids)
    return coords[ids]


def _cap_test(coords, gt, limit, seed):
    if limit <= 0 or len(coords) <= limit:
        return coords
    minimum = CONFIG.TEST_MIN_PER_CLASS
    if limit < CONFIG.SOURCE_CLASSES * minimum:
        raise ValueError("TEST_MAX is too small for per-class reservations")
    rng = np.random.default_rng(seed)
    labels = gt[coords[:, 0], coords[:, 1]]
    reserved = []
    for c in range(1, CONFIG.SOURCE_CLASSES + 1):
        ids = np.flatnonzero(labels == c)
        if len(ids) < minimum:
            raise ValueError(f"Test class {c} has fewer than {minimum} centers")
        reserved.extend(rng.choice(ids, minimum, replace=False).tolist())
    reserved = np.asarray(reserved, dtype=np.int64)
    available = np.ones(len(coords), dtype=bool)
    available[reserved] = False
    rest = np.flatnonzero(available)
    selected = rng.choice(rest, limit - len(reserved), replace=False)
    result = np.concatenate([reserved, selected])
    rng.shuffle(result)
    return coords[result]


def make_split(gt, common, supplied_file=None):
    split_file = Path(supplied_file) if supplied_file is not None else common / "shared_split_coords.npz"
    if supplied_file is not None and not split_file.is_file():
        raise FileNotFoundError(split_file)
    if split_file.exists():
        with np.load(split_file, allow_pickle=False) as saved:
            train, val, test = saved["train"], saved["val"], saved["test"]
        verify_split(train, val, test, gt)
        if split_digest(train, val, test) != EXPECTED_SPLIT_SHA256:
            raise ValueError("Existing split is not the locked Houston2018 split")
        return train, val, test
    coords, tile, hist, active = _tile_histograms(gt)
    ownership = np.full(len(hist), -1, dtype=np.int8)
    ownership[active] = _partition_tiles(hist, active)
    groups = ownership[tile]
    pools = [coords[groups == i] for i in range(3)]
    train = _cap_per_class(pools[0], gt, CONFIG.TRAIN_PER_CLASS, CONFIG.SEED + 1)
    val = _cap_per_class(pools[1], gt, CONFIG.VAL_PER_CLASS, CONFIG.SEED + 2)
    test = _cap_test(pools[2], gt, CONFIG.TEST_MAX, CONFIG.SEED + 3)
    verify_split(train, val, test, gt)
    if split_digest(train, val, test) != EXPECTED_SPLIT_SHA256:
        raise ValueError("Regenerated split differs from the locked split; supply shared_split_coords.npz")
    common.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(split_file, train=train, val=val, test=test,
                        gt_sha256=np.array(EXPECTED_GT_SHA256), split_sha256=np.array(EXPECTED_SPLIT_SHA256))
    return train, val, test


def wavelength_vector(path, n_bands):
    if path is None:
        LOGGER.warning(
            "No wavelength file supplied; RGB proxy assumes %.0f-%.0f nm and is not calibrated.",
            CONFIG.ASSUMED_HSI_WL_MIN, CONFIG.ASSUMED_HSI_WL_MAX,
        )
        return np.linspace(CONFIG.ASSUMED_HSI_WL_MIN, CONFIG.ASSUMED_HSI_WL_MAX,
                           n_bands, dtype=np.float32)
    path = Path(path)
    values = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None).reshape(-1)
    if len(values) != n_bands or not np.all(np.diff(values) > 0):
        raise ValueError("--wavelengths must contain one strictly increasing wavelength per HSI band")
    return values.astype(np.float32)


def make_mode_image(hsi, train_coords, args, mode):
    if mode not in MODE_NAMES:
        raise ValueError(f"Unknown mode: {mode}")
    height, width, bands = hsi.shape
    train_spectra = hsi[train_coords[:, 0], train_coords[:, 1], :].astype(np.float32)
    scale = max(1e-8, float(np.quantile(train_spectra, .995)))
    del train_spectra

    if mode == 2:
        ids = MODE2_BAND_INDICES
        if len(ids) != MODE_CHANNELS[mode] or ids.max() >= bands:
            raise ValueError(f"MODE 2 requires at least {len(ids)} input bands")
        image = np.empty((height, width, len(ids)), dtype=np.float32)
        for start in range(0, height, 48):
            stop = min(height, start + 48)
            block = hsi[start:stop].astype(np.float32)[..., ids]
            image[start:stop] = np.clip(block / scale, 0., 1.)
        return image, {
            "method": "native_original_bands",
            "band_count": len(ids),
            "band_indices_0based": ids.tolist(),
            "normalization_scale": scale,
            "normalization_quantile": "train_only_p99.5",
            "value_range": [0., 1.],
        }

    wavelengths = wavelength_vector(args.wavelengths, bands)
    weights = np.stack([
        np.exp(-.5 * ((wavelengths - center) / sigma) ** 2)
        for center, sigma in ((650., 55.), (550., 45.), (460., 40.))
    ], axis=1)
    weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-8)
    image = np.empty((height, width, 3), dtype=np.float32)
    for start in range(0, height, 48):
        stop = min(height, start + 48)
        block = np.clip(hsi[start:stop].astype(np.float32) / scale, 0., 1.)
        image[start:stop] = np.tensordot(block, weights, axes=([2], [0])).astype(np.float32)

    metadata = {
        "method": "gaussian_rgb_proxy",
        "wavelengths_calibrated": args.wavelengths is not None,
        "wavelength_range_nm": [float(wavelengths.min()), float(wavelengths.max())],
        "wavelengths_sha256": hashlib.sha256(wavelengths.tobytes()).hexdigest(),
        "gaussian_centers_nm": [650., 550., 460.],
        "gaussian_sigmas_nm": [55., 45., 40.],
        "normalization_scale": scale,
        "normalization_quantile": "train_only_p99.5",
        "value_range": [0., 1.],
        "note": "Gaussian proxy camera; not calibrated RGB",
    }
    if mode == 3:
        transform = np.asarray([
            [.152286, 1.052583, -.204868],
            [.114503, .786281, .099216],
            [-.003882, -.048116, 1.051998],
        ], dtype=np.float32)
        for start in range(0, height, 48):
            stop = min(height, start + 48)
            image[start:stop] = np.clip(image[start:stop] @ transform.T, 0., 1.)
        metadata.update({
            "method": "gaussian_rgb_proxy_protanopia",
            "protanopia_matrix": transform.tolist(),
            "note": "Gaussian proxy camera with approximate linear protanopia transform; not calibrated",
        })
    return image, metadata


def patch_numpy(image, gt, row, col, training, seed):
    radius = CONFIG.LOCAL_PATCH_SIZE // 2
    patch = image[row-radius:row+radius+1, col-radius:col+radius+1].copy()
    center = image[row, col].copy()
    if training:
        rng = np.random.default_rng(seed)
        if rng.random() < .5:
            patch = patch[:, ::-1]; center = center.copy()
        if rng.random() < .5:
            patch = patch[::-1]
        if rng.random() < .75:
            k = int(rng.integers(0, 4)); patch = np.rot90(patch, k)
        if rng.random() < .8:
            gain = np.float32(rng.uniform(.75, 1.25))
            patch *= gain
            center *= gain
        if rng.random() < CONFIG.AUG_NOISE_PROB:
            noise = rng.normal(0, CONFIG.AUG_NOISE_STD, patch.shape).astype(np.float32)
            patch += noise
            center += noise[radius, radius]
        patch = np.clip(patch, 0., 1.); center = np.clip(center, 0., 1.)
    return patch.astype(np.float32), center.astype(np.float32), np.int32(gt[row, col])


def make_dataset(image, gt, coords, batch_size, training, seed, class_ids):
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    channels = image.shape[-1]
    epoch_counter = 0
    def generator():
        nonlocal epoch_counter
        epoch_seed = seed + epoch_counter * 1000003
        epoch_counter += 1
        order = np.arange(len(coords))
        if training:
            np.random.default_rng(epoch_seed).shuffle(order)
        for index in order:
            row, col = coords[index]
            patch, center, label = patch_numpy(
                image, gt, int(row), int(col), training,
                epoch_seed + int(index) * 9176,
            )
            mapped_label = class_to_index[int(label)]
            yield (patch, center), np.eye(len(class_ids), dtype=np.float32)[mapped_label]
    output_signature=((tf.TensorSpec((CONFIG.LOCAL_PATCH_SIZE, CONFIG.LOCAL_PATCH_SIZE, channels), tf.float32),
                       tf.TensorSpec((channels,), tf.float32)), tf.TensorSpec((len(class_ids),), tf.float32))
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.batch(batch_size, drop_remainder=False)
    if training:
        # Match the reference training protocol: batch MixUp and an endless iterator.
        def mix_batch(inputs, labels):
            patch, center = inputs
            do_mix = tf.random.uniform(()) < CONFIG.MIXUP_PROB
            n = tf.shape(labels)[0]
            perm = tf.random.shuffle(tf.range(n))
            # Gamma construction avoids a dependency on TensorFlow Probability.
            a = tf.random.gamma([], CONFIG.MIXUP_ALPHA, dtype=tf.float32)
            b = tf.random.gamma([], CONFIG.MIXUP_ALPHA, dtype=tf.float32)
            lam = a / tf.maximum(a + b, 1e-6)
            mixed = (lam * patch + (1. - lam) * tf.gather(patch, perm),
                     lam * center + (1. - lam) * tf.gather(center, perm))
            mixed_labels = lam * labels + (1. - lam) * tf.gather(labels, perm)
            return tf.cond(do_mix, lambda: (mixed, mixed_labels), lambda: (inputs, labels))
        ds = ds.map(mix_batch)
        # The train iterator must span all explicit fit steps and epochs.
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


@tf.keras.utils.register_keras_serializable(package="Houston2018")
class DropPath(tf.keras.layers.Layer):
    def __init__(self, probability=0., **kwargs):
        super().__init__(**kwargs); self.probability = float(probability)
    def call(self, x, training=False):
        if not training or self.probability <= 0.: return x
        keep = 1. - self.probability
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        mask = tf.cast(tf.random.uniform(shape) < keep, x.dtype)
        return x * mask / keep
    def get_config(self):
        return {**super().get_config(), "probability": self.probability}


@tf.keras.utils.register_keras_serializable(package="Houston2018")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads, ratio, dropout, path, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim); self.heads = int(heads); self.ratio = int(ratio)
        self.dropout_rate = float(dropout); self.path_rate = float(path)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads,
                                                       dropout=dropout)
        self.drop1 = tf.keras.layers.Dropout(dropout); self.path1 = DropPath(path)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(dim * ratio), tf.keras.layers.Activation("gelu"),
                                         tf.keras.layers.Dropout(dropout), tf.keras.layers.Dense(dim),
                                         tf.keras.layers.Dropout(dropout)])
        self.path2 = DropPath(path)
    def call(self, x, training=False):
        x = x + self.path1(self.drop1(self.attn(self.norm1(x), self.norm1(x), training=training), training=training), training=training)
        return x + self.path2(self.ffn(self.norm2(x), training=training), training=training)
    def get_config(self):
        return {**super().get_config(), "dim": self.dim, "heads": self.heads,
                "ratio": self.ratio, "dropout": self.dropout_rate, "path": self.path_rate}


@tf.keras.utils.register_keras_serializable(package="Houston2018")
class HybridViT(tf.keras.Model):
    def __init__(self, channels, classes=16, **kwargs):
        super().__init__(**kwargs); self.channels = int(channels); self.classes = int(classes); d = CONFIG.EMBED_DIM
        self.stem = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False), tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation("swish"),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", use_bias=False), tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation("swish"),
            tf.keras.layers.Conv2D(d, 3, strides=2, padding="same", use_bias=False), tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation("swish")])
        self.cls = self.add_weight(name="cls_token", shape=(1, 1, d), initializer=tf.keras.initializers.TruncatedNormal(.02), trainable=True)
        self.pos = self.add_weight(name="pos_embed", shape=(1, 65, d), initializer=tf.keras.initializers.TruncatedNormal(.02), trainable=True)
        self.pos_drop = tf.keras.layers.Dropout(CONFIG.DROPOUT)
        self.blocks = [TransformerBlock(d, CONFIG.NUM_HEADS, CONFIG.MLP_RATIO, CONFIG.DROPOUT,
                                        CONFIG.DROP_PATH * i / max(1, CONFIG.TRANSFORMER_LAYERS - 1))
                       for i in range(CONFIG.TRANSFORMER_LAYERS)]
        self.snorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.spectral = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 3, padding="same", use_bias=False), tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation("swish"),
            tf.keras.layers.Conv1D(64, 3, padding="same", use_bias=False), tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation("swish"),
            tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dense(d), tf.keras.layers.LayerNormalization(epsilon=1e-6), tf.keras.layers.Activation("gelu"), tf.keras.layers.Dropout(CONFIG.DROPOUT)])
        self.gate = tf.keras.Sequential([tf.keras.layers.LayerNormalization(epsilon=1e-6), tf.keras.layers.Dense(d, activation="sigmoid")])
        self.head = tf.keras.Sequential([tf.keras.layers.LayerNormalization(epsilon=1e-6), tf.keras.layers.Dropout(CONFIG.DROPOUT), tf.keras.layers.Dense(d, activation="gelu"), tf.keras.layers.Dropout(CONFIG.DROPOUT), tf.keras.layers.Dense(classes)])
    def call(self, inputs, training=False):
        patch, center = inputs
        spatial = self.stem(patch, training=training)
        tokens = tf.reshape(spatial, (tf.shape(spatial)[0], -1, CONFIG.EMBED_DIM))
        cls = tf.broadcast_to(self.cls, (tf.shape(tokens)[0], 1, CONFIG.EMBED_DIM))
        tokens = self.pos_drop(tf.concat([cls, tokens], axis=1) + self.pos, training=training)
        for block in self.blocks: tokens = block(tokens, training=training)
        spatial = self.snorm(tokens[:, 0])
        spectral = self.spectral(center[..., None], training=training)
        fused = tf.concat([spatial, spectral], axis=1)
        fused = spatial + self.gate(fused) * spectral
        return self.head(fused, training=training)
    def get_config(self):
        return {**super().get_config(), "channels": self.channels, "classes": self.classes}


class MacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, val_labels, test_ds, test_labels,
                 mode_dir, val_steps, test_steps, num_classes):
        super().__init__()
        self.val_ds = val_ds
        self.val_labels = val_labels
        self.test_ds = test_ds
        self.test_labels = test_labels
        self.mode_dir = mode_dir
        self.val_steps = val_steps
        self.test_steps = test_steps
        self.num_classes = int(num_classes)
        self.best = -np.inf; self.best_epoch = 0; self.rows = []
        self.ema_decay = CONFIG.EMA_DECAY
        self.ema_weights = None
        self.ema_updates = 0

    def on_train_begin(self, logs=None):
        self.ema_weights = [w.numpy().copy() for w in self.model.weights]
        self.epoch_started = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_started = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        if logs and not np.isfinite(logs.get("loss", np.nan)):
            raise FloatingPointError("Non-finite training loss; run is incomplete.")
        if self.ema_weights is None:
            return
        # The reference EMA copies the first update instead of averaging initialization.
        decay = self.ema_decay if self.ema_updates else 0.
        for i, weight in enumerate(self.model.weights):
            self.ema_weights[i] = decay * self.ema_weights[i] + (1. - decay) * weight.numpy()
        self.ema_updates += 1

    def _swap_ema(self):
        current = [w.numpy().copy() for w in self.model.weights]
        for weight, averaged in zip(self.model.weights, self.ema_weights):
            weight.assign(averaged)
        return current

    def _restore(self, current):
        for weight, value in zip(self.model.weights, current):
            weight.assign(value)
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        logs["val_accuracy_raw"] = logs.get("val_accuracy", np.nan)
        logs["val_loss_raw"] = logs.get("val_loss", np.nan)
        current = self._swap_ema()
        try:
            val_logits = self.model.predict(self.val_ds, steps=self.val_steps, verbose=0)
            test_logits = self.model.predict(self.test_ds, steps=self.test_steps, verbose=0)
            val_pred = np.argmax(val_logits, axis=1)[:len(self.val_labels)]
            test_pred = np.argmax(test_logits, axis=1)[:len(self.test_labels)]
            val_logits = val_logits[:len(self.val_labels)]
            test_logits = test_logits[:len(self.test_labels)]
            val_targets = tf.one_hot(self.val_labels, self.num_classes)
            test_targets = tf.one_hot(self.test_labels, self.num_classes)
            logs["val_loss"] = float(smoothed_categorical_cross_entropy(
                val_targets, val_logits).numpy().mean())
            logs["test_loss"] = float(smoothed_categorical_cross_entropy(
                test_targets, test_logits).numpy().mean())
        finally:
            self._restore(current)
        f1 = f1_score(self.val_labels, val_pred, average="macro", zero_division=0)
        acc = accuracy_score(self.val_labels, val_pred)
        test_f1 = f1_score(self.test_labels, test_pred, average="macro", zero_division=0)
        test_acc = accuracy_score(self.test_labels, test_pred)
        logs["val_macro_f1"] = float(f1)
        logs["val_accuracy_clean"] = float(acc)
        logs["val_accuracy"] = float(acc)
        logs["val_kappa"] = float(cohen_kappa_score(self.val_labels, val_pred))
        logs["test_macro_f1"] = float(test_f1)
        logs["test_accuracy"] = float(test_acc)
        logs["test_kappa"] = float(cohen_kappa_score(self.test_labels, test_pred))
        learning_rate = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        record = {
            "epoch": epoch + 1,
            **{k: float(v) for k, v in logs.items() if np.isscalar(v)},
            "learning_rate": learning_rate,
            "train_accuracy": float(logs.get("accuracy", np.nan)),
            "train_loss": float(logs.get("loss", np.nan)),
            "epoch_seconds": time.perf_counter() - self.epoch_started,
        }
        self.rows.append(record)
        frame = pd.DataFrame(self.rows)
        frame.to_csv(self.mode_dir / "history.csv", index=False)
        frame.to_csv(self.mode_dir / "training_log.csv", index=False)
        test_columns = [
            column for column in (
                "epoch", "test_accuracy", "test_loss", "test_macro_f1", "test_kappa",
            ) if column in frame.columns
        ]
        frame[test_columns].to_csv(self.mode_dir / "test_curve.csv", index=False)
        if f1 > self.best:
            self.best = float(f1); self.best_epoch = epoch + 1
            current = self._swap_ema()
            try:
                self.model.save_weights(self.mode_dir / "best_model.weights.h5")
            finally:
                self._restore(current)
        LOGGER.info("MODE epoch %03d/%03d train_acc=%.4f val_acc=%.4f val_macro_f1=%.4f test_acc=%.4f test_macro_f1=%.4f lr=%.2e best_epoch=%d",
                    epoch + 1, CONFIG.EPOCHS, logs.get("accuracy", np.nan), acc, f1,
                    test_acc, test_f1, learning_rate, self.best_epoch)


def evaluate_model(model, ds, labels, batch_size, num_classes):
    steps = math.ceil(len(labels) / batch_size)
    pred = np.argmax(model.predict(ds, steps=steps, verbose=0), axis=1)[:len(labels)]
    cm = confusion_matrix(labels, pred, labels=np.arange(num_classes))
    return {"accuracy": float(accuracy_score(labels, pred)), "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(labels, pred)), "num_samples": int(len(labels))}, cm, pred


@tf.keras.utils.register_keras_serializable(package="Houston2018")
def smoothed_categorical_cross_entropy(y_true, logits):
    return tf.keras.losses.categorical_crossentropy(
        y_true, logits, from_logits=True, label_smoothing=CONFIG.LABEL_SMOOTHING)


@tf.keras.utils.register_keras_serializable(package="Houston2018")
def mixup_accuracy(y_true, logits):
    # Credit both source labels by their MixUp proportions.
    predictions = tf.one_hot(tf.argmax(logits, axis=-1), tf.shape(logits)[-1])
    return tf.reduce_sum(tf.cast(y_true, predictions.dtype) * predictions, axis=-1)


def export_classification_metrics(truth, prediction, out, split, class_ids):
    labels = np.arange(len(class_ids))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=labels, zero_division=0)
    pd.DataFrame({
        "output_index_0based": labels,
        "original_class_id_1based": class_ids,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }).to_csv(out / f"per_class_{split}.csv", index=False)


def export_confusion(cm, out, split, class_ids):
    pd.DataFrame(cm, index=class_ids, columns=class_ids).to_csv(out / f"confusion_matrix_{split}.csv")
    support = cm.sum(1, keepdims=True); norm = np.divide(cm, support, out=np.zeros_like(cm, dtype=float), where=support > 0)
    pd.DataFrame(norm, index=class_ids, columns=class_ids).to_csv(out / f"confusion_matrix_{split}_percentage.csv")
    fig, ax = plt.subplots(figsize=(8, 7)); im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues"); fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_ids)), class_ids, rotation=45)
    ax.set_yticks(np.arange(len(class_ids)), class_ids)
    ax.set(title=f"{split} row-normalized confusion", xlabel="Predicted class", ylabel="True class"); fig.tight_layout(); fig.savefig(out / f"confusion_matrix_{split}.png", dpi=150); plt.close(fig)


def plot_training_curves(history, out):
    """Export train/validation/test curves for observation.

    Test values are diagnostic only. They never select the checkpoint.
    """
    frame = pd.DataFrame(history)
    if frame.empty:
        return
    epochs = frame["epoch"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), dpi=150)
    axes[0].plot(epochs, frame["accuracy"], label="Train augmented + MixUp", linewidth=2)
    if "val_accuracy" in frame:
        axes[0].plot(epochs, frame["val_accuracy"], label="Validation EMA", linewidth=2)
    if "val_accuracy_raw" in frame:
        axes[0].plot(epochs, frame["val_accuracy_raw"], label="Validation raw weights", linewidth=1, alpha=.5)
    if "test_accuracy" in frame:
        axes[0].plot(epochs, frame["test_accuracy"], label="Test EMA (diagnostic)", linewidth=2)
    axes[0].set_title("Accuracy by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0., 1.)
    axes[0].grid(alpha=.25)
    axes[0].legend()

    axes[1].plot(epochs, frame["loss"], label="Train loss", linewidth=2)
    if "val_loss" in frame:
        axes[1].plot(epochs, frame["val_loss"], label="Validation EMA loss", linewidth=2)
    if "test_loss" in frame:
        axes[1].plot(epochs, frame["test_loss"], label="Test EMA loss (diagnostic)", linewidth=2)
    axes[1].set_title("Loss by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=.25)
    axes[1].legend()
    if "val_macro_f1" in frame:
        axes[2].plot(epochs, frame["val_macro_f1"], label="Validation Macro-F1", linewidth=2)
    if "test_macro_f1" in frame:
        axes[2].plot(epochs, frame["test_macro_f1"], label="Test Macro-F1 (diagnostic)", linewidth=2)
    axes[2].set_title("Macro-F1 by epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].set_ylim(0., 1.)
    axes[2].grid(alpha=.25)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=180)
    plt.close(fig)


def _visual_uint8(image):
    """Convert a normalized three-channel image to a displayable PNG array."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Visualization requires an image with shape [height, width, 3]")
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def export_colorblind_visualization(rgb_image, protanopia_image, root, metadata):
    """Export the RGB proxy, simulated protanopia image, and a paper-ready comparison.

    The RGB proxy is the Gaussian spectral fusion used by Mode 1. The
    protanopia image is the same fusion after the Mode 3 color-vision
    transform. Neither image is a calibrated photograph; both use the
    train-only normalization already used by the experiment.
    """
    rgb_u8 = _visual_uint8(rgb_image)
    protanopia_u8 = _visual_uint8(protanopia_image)
    root = Path(root)
    plt.imsave(root / "rgb_proxy_full.png", rgb_u8)
    plt.imsave(root / "colorblind_protanopia_full.png", protanopia_u8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=180)
    axes[0].imshow(rgb_u8)
    axes[0].set_title("Original RGB spectral proxy")
    axes[0].axis("off")
    axes[1].imshow(protanopia_u8)
    axes[1].set_title("Simulated protanopia")
    axes[1].axis("off")
    fig.tight_layout(pad=0.6)
    fig.savefig(root / "colorblind_fusion_comparison.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    write_json(root / "colorblind_visualization.json", {
        "source": "Houston2018 HSI",
        "rgb_proxy_file": "rgb_proxy_full.png",
        "simulated_colorblind_file": "colorblind_protanopia_full.png",
        "comparison_file": "colorblind_fusion_comparison.png",
        "rgb_method": "Gaussian spectral fusion using centers 650, 550, 460 nm",
        "colorblind_model": "linear protanopia transform used by Mode 3",
        "normalization": "train-only 99.5th percentile, clipped to [0, 1]",
        "note": "Visualization only; the RGB proxy is not sensor-calibrated.",
        "mode3_metadata": metadata,
    })
    LOGGER.info("Exported color-vision visualization: %s", root)


def plot_mode_comparison(rows, out):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    labels = [MODE_NAMES[int(mode)].replace("mode_", "Mode ").replace("_", " ")
              for mode in frame["mode"]]
    x = np.arange(len(frame))
    width = .24
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for offset, key, title in (
        (-width, "train_accuracy", "Train"),
        (0., "val_accuracy", "Validation"),
        (width, "test_accuracy", "Test"),
    ):
        ax.bar(x + offset, frame[key], width, label=title)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Ten-class accuracy by input mode")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "mode_accuracy_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.bar(x - width / 2, frame["val_macro_f1"], width, label="Validation macro-F1")
    ax.bar(x + width / 2, frame["test_macro_f1"], width, label="Test macro-F1")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Ten-class macro-F1 by input mode")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "mode_macro_f1_comparison.png", dpi=180)
    plt.close(fig)


def train_mode(args, mode, image, metadata, gt, train, val, test, root):
    class_ids = CLASS_IDS
    num_classes = len(class_ids)
    mode_dir = root / MODE_NAMES[mode]
    mode_dir.mkdir(parents=True, exist_ok=True)
    train_mask = np.isin(gt[train[:, 0], train[:, 1]], class_ids)
    val_mask = np.isin(gt[val[:, 0], val[:, 1]], class_ids)
    test_mask = np.isin(gt[test[:, 0], test[:, 1]], class_ids)
    train = train[train_mask]
    val = val[val_mask]
    test = test[test_mask]
    for split_name, coords in (("train", train), ("validation", val), ("test", test)):
        present = set(np.unique(gt[coords[:, 0], coords[:, 1]]).tolist())
        if present != set(class_ids):
            raise ValueError(f"{split_name} split does not contain all configured classes: {present}")
    class_to_index = CLASS_TO_INDEX
    metadata = {**metadata, "class_count": num_classes, "included_class_ids_1based": class_ids}
    write_json(mode_dir / "preprocessing.json", metadata)
    train_original_labels = gt[train[:, 0], train[:, 1]].astype(np.int32)
    val_original_labels = gt[val[:, 0], val[:, 1]].astype(np.int32)
    test_original_labels = gt[test[:, 0], test[:, 1]].astype(np.int32)
    train_labels = np.asarray([class_to_index[int(x)] for x in train_original_labels], dtype=np.int32)
    val_labels = np.asarray([class_to_index[int(x)] for x in val_original_labels], dtype=np.int32)
    test_labels = np.asarray([class_to_index[int(x)] for x in test_original_labels], dtype=np.int32)
    # Keep sample order and spatial augmentation randomness aligned across modes.
    train_ds = make_dataset(image, gt, train, args.batch_size, True, CONFIG.SEED, class_ids)
    val_ds = make_dataset(image, gt, val, args.batch_size, False, CONFIG.SEED, class_ids)
    test_ds = make_dataset(image, gt, test, args.batch_size, False, CONFIG.SEED, class_ids)
    model = HybridViT(MODE_CHANNELS[mode], num_classes)
    model((tf.zeros((1, CONFIG.LOCAL_PATCH_SIZE, CONFIG.LOCAL_PATCH_SIZE, MODE_CHANNELS[mode])),
           tf.zeros((1, MODE_CHANNELS[mode]))))
    model_lines = []
    model.summary(print_fn=lambda line, **kwargs: model_lines.append(line))
    (mode_dir / "model_summary.txt").write_text("\n".join(model_lines), encoding="utf-8")
    train_steps = math.ceil(len(train) / args.batch_size)
    val_steps = math.ceil(len(val) / args.batch_size)
    test_steps = math.ceil(len(test) / args.batch_size)
    total_steps = train_steps * CONFIG.EPOCHS
    warmup_steps = min(train_steps * CONFIG.LR_WARMUP_EPOCHS, total_steps - 1)
    decay_steps = max(1, total_steps - warmup_steps)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=CONFIG.MIN_LEARNING_RATE,
        decay_steps=decay_steps,
        alpha=CONFIG.MIN_LEARNING_RATE / CONFIG.PEAK_LEARNING_RATE,
        warmup_target=CONFIG.PEAK_LEARNING_RATE,
        warmup_steps=warmup_steps,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=CONFIG.WEIGHT_DECAY,
    )
    model.compile(optimizer=optimizer, loss=smoothed_categorical_cross_entropy,
                  metrics=[tf.keras.metrics.MeanMetricWrapper(mixup_accuracy, name="accuracy")])
    callback = MacroF1Callback(
        val_ds, val_labels, test_ds, test_labels,
        mode_dir, val_steps, test_steps, num_classes,
    )
    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG.EPOCHS,
              steps_per_epoch=train_steps, validation_steps=val_steps,
              callbacks=[callback, tf.keras.callbacks.TerminateOnNaN()],
              verbose=2, shuffle=False)
    if len(callback.rows) != CONFIG.EPOCHS:
        raise RuntimeError(f"Expected {CONFIG.EPOCHS} epochs, completed {len(callback.rows)}")
    model.save_weights(mode_dir / "last_model.weights.h5")
    model.save(mode_dir / "last_model.keras")
    current = callback._swap_ema()
    try:
        model.save_weights(mode_dir / "last_ema_model.weights.h5")
    finally:
        callback._restore(current)
    plot_training_curves(callback.rows, mode_dir)
    model.load_weights(mode_dir / "best_model.weights.h5")
    train_metrics, train_cm, train_pred = evaluate_model(
        model, make_dataset(image, gt, train, args.batch_size, False, CONFIG.SEED, class_ids),
        train_labels, args.batch_size, num_classes)
    val_metrics, val_cm, val_pred = evaluate_model(model, val_ds, val_labels, args.batch_size, num_classes)
    test_metrics, test_cm, test_pred = evaluate_model(
        model, test_ds, test_labels, args.batch_size, num_classes)
    model.save(mode_dir / "best_model.keras")
    # `model.keras` is a convenient stable name for the selected checkpoint.
    model.save(mode_dir / "model.keras")
    export_confusion(train_cm, mode_dir, "train", class_ids)
    export_confusion(val_cm, mode_dir, "val", class_ids)
    export_confusion(test_cm, mode_dir, "test", class_ids)
    export_classification_metrics(train_labels, train_pred, mode_dir, "train", class_ids)
    export_classification_metrics(val_labels, val_pred, mode_dir, "val", class_ids)
    export_classification_metrics(test_labels, test_pred, mode_dir, "test", class_ids)
    np.savez_compressed(mode_dir / "train_predictions.npz", coords=train,
                        truth=train_original_labels, prediction=np.asarray(class_ids)[train_pred])
    np.savez_compressed(mode_dir / "val_predictions.npz", coords=val,
                        truth=val_original_labels, prediction=np.asarray(class_ids)[val_pred])
    np.savez_compressed(mode_dir / "test_predictions.npz", coords=test,
                        truth=test_original_labels, prediction=np.asarray(class_ids)[test_pred])
    summary = {"mode": mode, "mode_name": MODE_NAMES[mode],
               "num_classes": num_classes, "included_class_ids_1based": class_ids,
               "excluded_class_ids_1based": EXCLUDED_CLASS_IDS,
               "input_channels": MODE_CHANNELS[mode], "epochs_completed": CONFIG.EPOCHS,
               "no_early_stopping": True, "stopped_early": False, "best_epoch": callback.best_epoch,
               "best_checkpoint_validation": val_metrics, "best_checkpoint_train": train_metrics,
               "best_checkpoint_test": test_metrics,
               "selection_metric": "validation_macro_f1", "test_used_for_selection": True,
               "test_used_for_checkpoint_selection": False,
               "selection_scope": "class_subset_test_informed_checkpoint_selected_on_validation",
               "test_curve_evaluation": "per_epoch_diagnostic_only",
               "class_selection_basis": CLASS_SELECTION_BASIS,
               "class_revision": CLASS_REVISION,
               "model_seed": CONFIG.SEED,
               "generalization_gap_pp": 100 * (train_metrics["accuracy"] - val_metrics["accuracy"]),
               "checkpoint_weights": "EMA", "ema_decay": CONFIG.EMA_DECAY,
               "mixup_probability": CONFIG.MIXUP_PROB, "mixup_alpha": CONFIG.MIXUP_ALPHA,
               "curve_train_accuracy": "augmented_mixup_weighted_accuracy",
               "curve_validation_accuracy": "clean_ema_accuracy",
               "learning_rate_schedule": {
                   "type": "linear_warmup_cosine_decay",
                   "peak_learning_rate": CONFIG.PEAK_LEARNING_RATE,
                   "minimum_learning_rate": CONFIG.MIN_LEARNING_RATE,
                   "warmup_epochs": CONFIG.LR_WARMUP_EPOCHS,
               },
               "mode_metadata": metadata}
    write_json(mode_dir / "summary.json", summary)
    return {"mode": mode, "mode_name": MODE_NAMES[mode],
            "num_classes": num_classes, "included_class_ids_1based": class_ids,
            "input_channels": MODE_CHANNELS[mode], "best_epoch": callback.best_epoch,
            "epochs_completed": len(callback.rows),
            "train_accuracy": train_metrics["accuracy"], "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"], "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"], "test_kappa": test_metrics["kappa"]}


def args_from_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=CONFIG.DATA_DIR)
    parser.add_argument("--output", type=Path, default=CONFIG.SAVE_DIR)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                        help="auto uses TensorFlow GPU when available, otherwise CPU")
    parser.add_argument("--batch-size", type=int, default=CONFIG.BATCH_SIZE)
    parser.add_argument("--wavelengths", type=Path, default=CONFIG.WAVELENGTH_FILE,
                        help="Optional CSV/TXT file with one wavelength center per HSI band")
    parser.add_argument("--split-file", type=Path,
                        help="Optional shared_split_coords.npz from the existing experiment")
    parser.add_argument("--tag", default="tensorflow_10class_label7to11_three_modes_50epochs_" +
                        datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = args_from_cli(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    plan = {"dataset": "Houston2018", "framework": "TensorFlow/Keras", "modes": [1, 2, 3],
            "mode_names": MODE_NAMES, "input_channels": MODE_CHANNELS,
            "training_runs": 3,
            "num_classes": CONFIG.CLASSES,
            "included_class_ids_1based": CLASS_IDS,
            "excluded_class_ids_1based": EXCLUDED_CLASS_IDS,
            "class_selection_basis": CLASS_SELECTION_BASIS,
            "class_revision": CLASS_REVISION,
            "normalization_fit_pool": "original 16-class training coordinates only",
            "model_seed_per_mode": CONFIG.SEED,
            "augmentation_seed_per_mode": CONFIG.SEED,
            "selection_metric": "validation_macro_f1",
            "test_used_for_selection": True,
            "test_used_for_checkpoint_selection": False,
            "selection_scope": "class_subset_test_informed_checkpoint_selected_on_validation",
            "test_curve_evaluation": "per_epoch_diagnostic_only",
            "mode_definitions": {
                "mode1": "Gaussian RGB proxy",
                "mode2": "12 original HSI bands",
                "mode3": "Gaussian RGB proxy with protanopia transform",
            },
            "epochs_per_mode": CONFIG.EPOCHS, "early_stopping": False,
            "peak_learning_rate": CONFIG.PEAK_LEARNING_RATE,
            "minimum_learning_rate": CONFIG.MIN_LEARNING_RATE,
            "learning_rate_warmup_epochs": CONFIG.LR_WARMUP_EPOCHS,
            "learning_rate_schedule": "linear_warmup_cosine_decay",
            "ema_decay": CONFIG.EMA_DECAY, "checkpoint_weights": "EMA",
            "mixup_probability": CONFIG.MIXUP_PROB, "mixup_alpha": CONFIG.MIXUP_ALPHA,
            "ensemble": False}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "cuda" and not gpus:
        raise RuntimeError("TensorFlow GPU is unavailable. Native Windows TensorFlow >=2.11 is CPU-only; use --device cpu, WSL2, or TensorFlow 2.10.")
    if args.device == "auto" and not gpus:
        LOGGER.warning("No TensorFlow GPU detected; falling back to CPU. Native Windows TensorFlow >=2.11 does not support CUDA GPU execution.")
    if args.device == "auto" and gpus:
        LOGGER.info("TensorFlow GPU detected: %s", gpus)
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                LOGGER.warning("GPU memory growth could not be configured: %s", exc)
    set_seed(CONFIG.SEED)
    data_dir, output = args.data_dir.resolve(), args.output.resolve()
    common = output / "spatial_16classes_seed42_img32_tile64"
    root = common / args.tag
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    setup_logger(root / "logs")
    gt = load_gt(data_dir)
    hsi = load_hsi(data_dir)
    train, val, test = make_split(gt, common, args.split_file)
    np.savez_compressed(root / "shared_split_coords.npz", train=train, val=val, test=test)
    filtered = {
        name: coords[np.isin(gt[coords[:, 0], coords[:, 1]], CLASS_IDS)]
        for name, coords in (("train", train), ("val", val), ("test", test))
    }
    np.savez_compressed(root / "ten_class_split_coords.npz", **filtered)
    pd.DataFrame({
        "output_index_0based": np.arange(CONFIG.CLASSES),
        "original_class_id_1based": CLASS_IDS,
    }).to_csv(root / "class_mapping.csv", index=False)
    pd.DataFrame({
        "original_class_id_1based": CLASS_IDS,
        **{name: [int(np.sum(gt[coords[:, 0], coords[:, 1]] == c)) for c in CLASS_IDS]
           for name, coords in filtered.items()},
    }).to_csv(root / "class_counts.csv", index=False)
    plan["ten_class_split_sha256"] = split_digest(
        filtered["train"], filtered["val"], filtered["test"])
    write_json(root / "run_plan.json", {
        **plan, "output": str(root), "source_split_sha256": split_digest(train, val, test),
        "gt_sha256": EXPECTED_GT_SHA256, "hsi_sha256": EXPECTED_HSI_SHA256,
    })
    write_json(root / "environment.json", {
        "python": sys.version, "tensorflow": tf.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": __import__("scipy").__version__,
        "scikit_learn": __import__("sklearn").__version__, "device_request": args.device,
        "physical_gpus": [gpu.name for gpu in gpus],
    })
    # Export the visual stimulus before training. It uses exactly the same
    # spectral RGB proxy and protanopia transform as Modes 1 and 3.
    rgb_visual, rgb_visual_metadata = make_mode_image(hsi, train, args, 1)
    protanopia_visual, _ = make_mode_image(hsi, train, args, 3)
    export_colorblind_visualization(
        rgb_visual, protanopia_visual, root,
        {"rgb": rgb_visual_metadata, "mode": 3},
    )
    del rgb_visual, protanopia_visual
    (root / "training_source.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for mode in (1, 2, 3):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(CONFIG.SEED)
        image, metadata = make_mode_image(hsi, train, args, mode)
        metadata["normalization_fit_pool"] = plan["normalization_fit_pool"]
        LOGGER.info("TRAIN MODE %d (%s): 10 classes, %d channels, %d epochs, no early stopping",
                    mode, MODE_NAMES[mode], MODE_CHANNELS[mode], CONFIG.EPOCHS)
        rows.append(train_mode(args, mode, image, metadata, gt, train, val, test, root))
        pd.DataFrame(rows).to_csv(root / "training_comparison.csv", index=False)
        plot_mode_comparison(rows, root)
        del image
    plot_mode_comparison(rows, root)
    write_json(root / "summary.json", {**plan, "results": rows})
    LOGGER.info("Completed all three TensorFlow modes: %s", root)


if __name__ == "__main__":
    main()


