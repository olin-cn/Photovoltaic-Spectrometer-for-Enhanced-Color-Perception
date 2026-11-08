# Photovoltaic Spectrometer for Enhanced Color Perception

This repository contains three workflows supporting the vdWH photovoltaic spectrometer study:

1. Monochromatic and dual-peak spectral reconstruction from measured photovoltaic responses (MATLAB).
2. Broadband spectral reconstruction from measured photovoltaic responses (Python).
3. Houston2018 hyperspectral image classification using RGB proxy, native hyperspectral, and simulated protanopia inputs (TensorFlow/Keras).

The reconstruction workflows use the supplied measured device-response data. The classification workflow uses Houston2018 HSI measurements with three input representations and a common HybridViT classifier.

---

## 1. Repository structure

```text
.
├── 1.monochromatic_dual-peak reconstruction/
│   ├── Reconstruction_Main.m
│   ├── data/
│   │   ├── ResponseMatrix_1.xlsx
│   │   ├── ResponseMatrix_2.xlsx
│   │   ├── MeasuredSignals.xlsx
│   │   ├── Fig3cSpectra.xlsx
│   │   └── Fig3hSpectra.xlsx
│   └── outputs/                         # generated
│
├── 2.broadband reconstruction/
│   ├── main.py
│   ├── ResponseMatrix.txt
│   ├── MeasuredSignals_1.txt
│   ├── MeasuredSignals_2.txt
│   ├── Refspectra_1.txt
│   └── Refspectra_2.txt
│
├── 3. Image classification/
│   ├── Train.py
│   ├── data/
│   │   └── Houston2018/
│   │       ├── houston_hsi.mat
│   │       └── houston_gt.mat
│   └── results_houston_tensorflow/      # generated
│
└── README.md
```

The released classification script is named **`Train.py`**. The input and output paths used by this script are resolved relative to its location by default.

## 2. Software requirements

- MATLAB R2020b or later is recommended for `Reconstruction_Main.m`; its local nonnegative quadratic-programming solver does not require the Optimization Toolbox.
- Python, NumPy, SciPy, Pandas, Matplotlib, scikit-learn, and TensorFlow/Keras are used by the Python workflows.
- `scipy.optimize.milp` is used for spatial-split generation (SciPy 1.9+). TensorFlow 2.18.0 is a suitable version for the current training API.

```bash
python -m pip install numpy scipy pandas matplotlib scikit-learn tensorflow tqdm
```

`Train.py` supports TensorFlow CPU and GPU execution. On native Windows, TensorFlow 2.11+ requires an alternative supported environment such as WSL2 for CUDA GPU execution.

---

## 3. File naming

The MATLAB reconstruction reads `ResponseMatrix_1.xlsx` and `ResponseMatrix_2.xlsx` from its `data/` directory. The broadband reconstruction uses `Refspectra_1.txt` and `Refspectra_2.txt` with the capitalization shown above. On case-sensitive systems, filenames should match exactly.

---

# Part I. Monochromatic and dual-peak spectral reconstruction

## 4. Purpose

The MATLAB workflow reconstructs spectra from fixed measured photovoltaic response vectors using measured response matrices, a Gaussian spectral dictionary, nonnegative quadratic optimization, and regularization.

The forward relationship is:

```text
MeasuredSignal(sensor)
    = integral[
        Spectrum(wavelength)
        × ResponseMatrix(wavelength, sensor)
      ] d(wavelength)
```

The reconstruction script solves the corresponding inverse problem.

## 5. Entry point

```text
1.monochromatic_dual-peak reconstruction/Reconstruction_Main.m
```

The script resolves all input and output paths relative to its own location:

```matlab
packageDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(packageDir, 'data');
outDir = fullfile(packageDir, 'outputs');
```

The user therefore does not need to manually edit absolute paths.

## 6. Required input data

| File | Sheet | Purpose |
|---|---|---|
| `data/ResponseMatrix_1.xlsx` | first/default sheet | measured response matrix used for Fig. 3c |
| `data/ResponseMatrix_2.xlsx` | first/default sheet | measured response matrix used for Fig. 3h |
| `data/MeasuredSignals.xlsx` | `Fig3c` | measured response vectors for the Fig. 3c spectra |
| `data/MeasuredSignals.xlsx` | `Fig3h` | measured response vector for Fig. 3h |
| `data/Fig3cSpectra.xlsx` | `ReferenceSpectra` | wavelength axis and reference spectra used for plotting |
| `data/Fig3hSpectra.xlsx` | `ReferenceSpectra` | wavelength axis and target spectrum used for plotting |

The response-matrix spreadsheets are expected to contain wavelength values in the first column and sensor responses in the remaining columns. The first row/column are removed as implemented in `Reconstruction_Main.m`.

## 7. Reconstruction parameters

| Parameter | Fig. 3c | Fig. 3h |
|---|---:|---:|
| Response matrix | `ResponseMatrix_1.xlsx` | `ResponseMatrix_2.xlsx` |
| Number of interpolated wavelength points | 641 | 641 |
| Number of Gaussian basis functions | 641 | 641 |
| Gaussian-basis FWHM | 10.0 nm | 0.5 nm |
| Nonnegative solution | yes | yes |
| Integration rule | basis integration | basis integration |
| Zero-order penalty coefficient | 0 | 0 |
| Second-order penalty coefficient | `1e-5` | `1e-4` |
| Added noise in reproduction script | none | none |
| Output normalization | divide by maximum | divide by maximum |

The script uses measured signals loaded from the supplied Excel file. It does not synthesize or replace those signals during reconstruction.

## 8. How to run

### MATLAB graphical interface

1. Open MATLAB.
2. Navigate to `1.monochromatic_dual-peak reconstruction`.
3. Run:

```matlab
Reconstruction_Main
```

### MATLAB command line

From the repository root:

```bash
matlab -batch "run('1.monochromatic_dual-peak reconstruction/Reconstruction_Main.m')"
```

## 9. Generated outputs

The script creates `outputs/` automatically and writes:

```text
outputs/Fig3c_Reconstruction.csv
outputs/Fig3c_Reconstruction.png
outputs/Fig3h_Reconstruction.csv
outputs/Fig3h_Reconstruction.png
outputs/ReproductionResults_vdwRS2_loaded.mat
```

The CSV files contain the reconstructed wavelength axis and normalized reconstructed curves. The MAT file stores the reconstruction and measurement structures used by the script.

## 10. Figure-numbering note

The supplied MATLAB package currently labels the second reconstruction as **Fig. 3h**. If the corresponding panel is renumbered as **Fig. 3g** in the revised manuscript, the figure number must be changed consistently in:

- the manuscript;
- `Reconstruction_Main.m`;
- the Excel sheet and file names, if renamed;
- generated output names; and
- this README.

The data and panel identity should be verified before renaming; only the label should change, not the underlying dataset.

---

# Part II. Broadband spectral reconstruction

## 11. Purpose

The Python workflow reconstructs two representative broadband spectra from measured photovoltaic response vectors using:

1. row-normalized response matrices;
2. a Gaussian spectral basis;
3. generalized cross-validation to select the regularization parameter; and
4. nonnegative least squares.

## 12. Entry point and required files

```text
2.broadband reconstruction/main.py
```

Required files in the same folder:

```text
ResponseMatrix.txt
MeasuredSignals_1.txt
MeasuredSignals_2.txt
Refspectra_1.txt
Refspectra_2.txt
```

## 13. Numerical configuration

| Parameter | Value |
|---|---:|
| Wavelength range | 360–1000 nm |
| Number of wavelength samples | 65 |
| Number of Gaussian basis functions | 65 |
| Regularization selection | generalized cross-validation |
| Constrained solver | SciPy `nnls` |
| Response-matrix normalization | each row divided by its row maximum |
| Reconstructed-spectrum normalization | divide by maximum |

The Gaussian-basis FWHM settings are:

| Dataset | Ideal input | Experimental input |
|---|---:|---:|
| Dataset 1 | 10 nm | 60 nm |
| Dataset 2 | 10 nm | 120 nm |

For each dataset, column 1 of `MeasuredSignals_*.txt` is treated as the ideal response and column 2 as the experimental response. The first column of each `Refspectra_*.txt` file is the wavelength axis; the remaining column(s) contain reference spectra.

## 14. How to run

Because the current `main.py` reads its input files relative to the working directory, first change into the workflow folder.

### Windows PowerShell

```powershell
cd ".\2.broadband reconstruction"
python .\main.py
```

### Linux/macOS

```bash
cd "2.broadband reconstruction"
python main.py
```

## 15. Generated outputs

For each run, the script displays two reconstruction figures and writes timestamped text files:

```text
recon_1_YYYYMMDD_HHMMSS.txt
recon_2_YYYYMMDD_HHMMSS.txt
```

Each output file contains:

```text
wavelength_nm, reconstructed_ideal, reconstructed_experimental
```

---

# Part III. Houston2018 hyperspectral image classification

## 16. Purpose and entry point

```text
3. Image classification/Train.py
```

A run trains **three independent ten-class HybridViT models**, one per input mode, using the same spatial data split. Modes run consecutively in the order 1, 2, 3.

| Mode | Input representation | Channels | Result directory |
|---:|---|---:|---|
| 1 | Gaussian-fused RGB proxy from HSI | 3 | `mode_1_rgb/` |
| 2 | Twelve selected original HSI bands | 12 | `mode_2_native/` |
| 3 | RGB proxy with a simulated protanopia transformation | 3 | `mode_3_protanopia/` |

## 17. Input data and fixed classes

Place the input files beside `Train.py` in `data/Houston2018/`:

```text
data/Houston2018/
├── houston_hsi.mat
└── houston_gt.mat
```

The script loads the MATLAB variables `houston_hsi` (a 50-band image cube) and `houston_gt` (the ground-truth labels). The experiment uses ten original Houston2018 class IDs:

```python
CLASS_IDS = [1, 2, 3, 4, 8, 9, 11, 14, 15, 16]
```

They map in the given order to model output indices `0–9`. Training labels come from the **center pixel** of each spatial patch; exported predictions and confusion matrices use the original class IDs.

The ten-class subset replaces class 7 with class 11 based on train/validation spatial-tile coverage following inspection of earlier Mode-2 test results. Results on the existing test split are exploratory.

The script verifies fixed SHA-256 hashes for the ground-truth array, raw HSI array, and shared split coordinates:

| Object | Expected SHA-256 |
|---|---|
| Ground truth (filtered `uint8`) | `3d287d75afc131717b634179ae4476e7d0c8cd3e1149ee8282a80a03f4ce5910` |
| Raw HSI data | `84588a62ca5feb936ed45e17566be9721ce2620bef3585ec299bb4b4aa5dcb4d` |
| Spatial split | `e91e11f08255d957b57ff92a91ae2b3fea60c9ced2bf7b9e40d9173f96bbe46f` |

## 18. Spatial train/validation/test split

A shared spatial split is built on all 16 original classes before filtering to the selected ten. Eligible sample centers are grouped into **64 × 64-pixel tiles** and the tiles are partitioned using SciPy mixed-integer linear programming (`milp`). Training, validation, and test tiles are disjoint, and sample neighborhoods remain inside their assigned tile.

| Parameter | Value |
|---|---:|
| Spatial tile size | 64 × 64 |
| Eligibility window (`IMG_SIZE`) | 32 × 32 |
| Extracted training/evaluation patch | 31 × 31 |
| Training sample cap | 1,000 per original class |
| Validation sample cap | 250 per original class |
| Test sample cap | 40,000 total |
| Minimum reserved test samples per original class if capped | 10 |
| Minimum per-class region count | 10 |
| MILP time limit | 60 s |
| Random seed | 42 |

The common split is stored at:

```text
results_houston_tensorflow/spatial_16classes_seed42_img32_tile64/shared_split_coords.npz
```

On startup, `Train.py` loads that split if present or attempts to generate it. It checks the coordinate digest against the fixed reference. An existing matching split can also be supplied using `--split-file`.

## 19. Input preprocessing

### Mode 1: RGB proxy

The original 50 HSI bands are combined with normalized Gaussian spectral weights to produce RGB channels:

| Channel | Center | Gaussian σ |
|---|---:|---:|
| Red | 650 nm | 55 nm |
| Green | 550 nm | 45 nm |
| Blue | 460 nm | 40 nm |

A supplied 50-element, strictly increasing wavelength vector (`--wavelengths`) defines the band centers. By default, the code uses an assumed linearly spaced **380–1000 nm** axis.

### Mode 2: Native HSI bands

Mode 2 takes 12 channels directly from the original HSI cube, using these zero-based band indices:

```python
MODE2_BAND_INDICES = [0, 4, 9, 13, 18, 22, 27, 31, 36, 40, 45, 49]
```

These are native HSI measurements, with no PCA or RGB-to-spectrum conversion.

### Mode 3: Simulated protanopia

Mode 3 applies the following matrix to the RGB proxy and clips the result to `[0, 1]`:

```text
[[ 0.152286,  1.052583, -0.204868],
 [ 0.114503,  0.786281,  0.099216],
 [-0.003882, -0.048116,  1.051998]]
```

### Common intensity scaling

A single scaling factor is calculated as the **99.5th percentile of spectra from the original 16-class training coordinates**, and all three image representations are scaled and clipped to `[0, 1]`. Validation and test coordinates are excluded from estimating the factor. Each mode saves its preprocessing metadata to `preprocessing.json`.

## 20. Samples and augmentation

Each sample contains a `31 × 31 × C` spatial patch and its center-pixel vector of length `C` (with `C = 3` or `12`). The center-pixel label determines the ten-class target.

Only the training pipeline applies random augmentation:

| Operation | Configuration |
|---|---|
| Horizontal flip | Probability 0.5 |
| Vertical flip | Probability 0.5 |
| Rotation by multiples of 90° | Probability 0.75 |
| Multiplicative intensity gain | Probability 0.8; uniform 0.75–1.25 |
| Gaussian noise | Probability 0.5; σ = 0.01 |
| Batch MixUp | Probability 0.2; Beta(0.2, 0.2) mixing weights |

Training samples are shuffled for each epoch. Validation and test samples are evaluated without augmentation.

## 21. HybridViT architecture

The same **two-branch HybridViT** is used in all modes:

1. A three-layer convolutional stem transforms the spatial patch into an `8 × 8` feature map (64 spatial tokens).
2. A learned CLS token and positional embeddings produce 65 Transformer tokens.
3. Three Transformer blocks extract spatial context.
4. A separate 1D-convolutional spectral branch processes the center-pixel vector.
5. A learned sigmoid gate fuses spatial and spectral features, followed by a ten-logit classification head.

| Model setting | Value |
|---|---:|
| Stem output channels | 32, 64, 96 |
| Embedding dimension | 96 |
| Transformer blocks | 3 |
| Attention heads | 4 |
| MLP ratio | 3 |
| Dropout | 0.20 |
| Maximum DropPath rate | 0.10 |
| Output classes | 10 |

The network produces logits; categorical cross-entropy is computed with `from_logits=True`.

## 22. Training configuration

Each mode trains for **50 epochs**, without early stopping, using independent weights and aligned random seeds.

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Epochs | 50 |
| Batch size | 32 |
| Optimizer | AdamW |
| Peak learning rate | `5e-5` |
| Minimum learning rate | `1e-6` |
| Learning-rate schedule | 3-epoch warmup, then cosine decay |
| Weight decay | `1e-2` |
| Label smoothing | 0.05 |
| Loss | Smoothed categorical cross-entropy |
| EMA decay | 0.99 |
| Checkpoint criterion | Validation macro-F1 with EMA weights |

At the end of each epoch, validation and test metrics are logged using EMA weights. **The best checkpoint is selected by validation macro-F1.** The selected checkpoint is then evaluated on the unaugmented training, validation, and test samples. Reported metrics include overall accuracy, macro-F1, and Cohen’s kappa; per-class precision, recall, F1, and support are also exported.

## 23. Run instructions

From the repository root, move to the classification directory and run `Train.py`:

```bash
cd "3. Image classification"
python Train.py
```

Windows PowerShell:

```powershell
cd ".\3. Image classification"
python .\Train.py
```

To print the configuration without starting training:

```bash
python Train.py --dry-run
```

Example with explicit inputs:

```bash
python Train.py \
  --data-dir ./data/Houston2018 \
  --split-file ./shared_split_coords.npz \
  --wavelengths ./wavelengths.csv \
  --device auto \
  --batch-size 32 \
  --tag houston_run01
```

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `data/Houston2018/` | HSI and ground-truth MAT files |
| `--output` | `results_houston_tensorflow/` | Results directory |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--batch-size` | `32` | Batch size |
| `--wavelengths` | none | Optional 50-element wavelength vector for Modes 1 and 3 |
| `--split-file` | none | Existing matching spatial split |
| `--tag` | Timestamped name | Unique run identifier |
| `--dry-run` | off | Display experiment plan |

The default run executes all three modes. Use a new `--tag` for each run; existing run folders are not overwritten.

## 24. Output files

The classification script writes a run directory within `results_houston_tensorflow/spatial_16classes_seed42_img32_tile64/`:

```text
results_houston_tensorflow/
└── spatial_16classes_seed42_img32_tile64/
    ├── shared_split_coords.npz            # saved here when generated
    └── <tag>/
        ├── logs/run.log
        ├── shared_split_coords.npz
        ├── ten_class_split_coords.npz
        ├── class_mapping.csv
        ├── class_counts.csv
        ├── run_plan.json
        ├── environment.json
        ├── training_source.py
        ├── rgb_proxy_full.png
        ├── colorblind_protanopia_full.png
        ├── colorblind_fusion_comparison.png
        ├── colorblind_visualization.json
        ├── training_comparison.csv
        ├── mode_accuracy_comparison.png
        ├── mode_macro_f1_comparison.png
        ├── summary.json
        ├── mode_1_rgb/
        ├── mode_2_native/
        └── mode_3_protanopia/
```

Every `mode_*` folder contains:

```text
preprocessing.json
model_summary.txt
best_model.weights.h5
best_model.keras
model.keras
last_model.weights.h5
last_model.keras
last_ema_model.weights.h5
history.csv
training_log.csv
test_curve.csv
training_curves.png
confusion_matrix_train.csv
confusion_matrix_train_percentage.csv
confusion_matrix_train.png
confusion_matrix_val.csv
confusion_matrix_val_percentage.csv
confusion_matrix_val.png
confusion_matrix_test.csv
confusion_matrix_test_percentage.csv
confusion_matrix_test.png
per_class_train.csv
per_class_val.csv
per_class_test.csv
train_predictions.npz
val_predictions.npz
test_predictions.npz
summary.json
```

`best_model.keras` and `model.keras` contain the selected EMA checkpoint. `last_model.keras` contains the last-epoch raw model. Prediction archives contain pixel coordinates and original class IDs; confusion matrices include both counts and row-normalized values. `training_comparison.csv` summarizes the three modes.

---

# Part IV. Figure reproduction map

## 25. Fig. 3c — monochromatic/dual-peak reconstruction

```text
Entry point: 1.monochromatic_dual-peak reconstruction/Reconstruction_Main.m
Inputs:      data/ResponseMatrix_1.xlsx
             data/MeasuredSignals.xlsx (Fig3c)
             data/Fig3cSpectra.xlsx (ReferenceSpectra)
Parameters:  641 wavelength points and Gaussian basis functions
             FWHM = 10 nm; second-order penalty = 1e-5
Outputs:     outputs/Fig3c_Reconstruction.csv
             outputs/Fig3c_Reconstruction.png
```

## 26. Fig. 3h — spectral reconstruction

```text
Entry point: 1.monochromatic_dual-peak reconstruction/Reconstruction_Main.m
Inputs:      data/ResponseMatrix_2.xlsx
             data/MeasuredSignals.xlsx (Fig3h)
             data/Fig3hSpectra.xlsx (ReferenceSpectra)
Parameters:  641 wavelength points and Gaussian basis functions
             FWHM = 0.5 nm; second-order penalty = 1e-4
Outputs:     outputs/Fig3h_Reconstruction.csv
             outputs/Fig3h_Reconstruction.png
```

## 27. Houston2018 three-mode classification comparison

```text
Entry point: 3. Image classification/Train.py
Dataset:     Houston2018, ten retained classes
Modes:       RGB proxy / 12 native HSI bands / simulated protanopia
Backbone:    Centered spatial–spectral HybridViT
Training:    50 epochs per mode, EMA checkpoint selected by validation macro-F1
Outputs:     training_comparison.csv
             mode_accuracy_comparison.png
             mode_macro_f1_comparison.png
             per-mode metrics and confusion matrices
```

---

## 28. Citation

If this repository is used in academic work, please cite the corresponding manuscript.

```text
Citation information will be added after publication.
```

## 29. Contact and code-use policy

This repository is provided for academic reproducibility and methodological review. For questions about the data, code, permissions, or collaboration, contact the corresponding author. Use and redistribution should follow the license or permission terms supplied with the repository.
