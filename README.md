# Photovoltaic Spectrometer for Enhanced Color Perception

This repository contains three workflows supporting the vdWH photovoltaic spectrometer study:

1. Monochromatic and dual-peak spectral reconstruction from measured photovoltaic responses (MATLAB).
2. Broadband spectral reconstruction from measured photovoltaic responses (Python).
3. Houston2018 hyperspectral image classification using RGB proxy, native hyperspectral, and simulated protanopia inputs (TensorFlow/Keras).

The reconstruction workflows use measured device-response data. The classification workflow uses Houston2018 hyperspectral measurements with three input representations and a common HybridViT classifier.

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
│   ├── refspectra_1.txt
│   ├── refspectra_2.txt
│   ├── Raw_data/
│   │   ├── measuredsignals_1.csv
│   │   ├── measuredsignals_2.csv
│   │   └── reference_spectra_1.txt
│   ├── recon_1_YYYYMMDD_HHMMSS.txt     # generated
│   └── recon_2_YYYYMMDD_HHMMSS.txt     # generated
│
├── 3. Image classification/
│   ├── Train.py
│   ├── data/
│   │   └── Houston2018/
│   │       ├── houston_hsi.mat          # download separately
│   │       └── houston_gt.mat           # download separately
│   └── results_houston_tensorflow/      # generated
│
└── README.md
```

The Houston2018 data files are downloaded separately because of repository storage limits. The classification entry script is named **`Train.py`**. All three workflows resolve their input paths relative to their own script locations.

## 2. Software requirements

- MATLAB R2020b or later is recommended for `Reconstruction_Main.m`. The script includes a local nonnegative quadratic-programming solver and does not require the Optimization Toolbox.
- Python 3.9 or later, NumPy, SciPy, Pandas, Matplotlib, scikit-learn, and TensorFlow/Keras support the Python workflows.
- SciPy 1.9 or later is required for `scipy.optimize.milp` in the classification spatial split. TensorFlow 2.18.0 is compatible with the training API used by `Train.py`.

```bash
python -m pip install numpy scipy pandas matplotlib scikit-learn tensorflow
```

`Train.py` supports CPU and GPU execution. On native Windows, recent TensorFlow versions support CUDA GPU execution through environments such as WSL2.

## 3. File naming and paths

`Reconstruction_Main.m` reads the two `ResponseMatrix_*.xlsx` files from its `data/` directory. The broadband `main.py` uses the exact lowercase names `refspectra_1.txt` and `refspectra_2.txt` in its `ReferenceFiles` mapping. Filenames must match the script, including capitalization on case-sensitive systems.

---

# Part I. Monochromatic and dual-peak spectral reconstruction

## 4. Purpose

The MATLAB workflow reconstructs monochromatic and dual-peak spectra from measured photovoltaic response vectors using measured response matrices, a Gaussian spectral dictionary, regularization, and nonnegative quadratic optimization.

The forward relationship is:

```text
MeasuredSignal(sensor)
    = integral[
        Spectrum(wavelength)
        × ResponseMatrix(wavelength, sensor)
      ] d(wavelength)
```

The script solves the corresponding inverse problem for the Fig. 3c and Fig. 3h measurements.

## 5. Entry point

```text
1.monochromatic_dual-peak reconstruction/Reconstruction_Main.m
```

The script sets its paths relative to its own location:

```matlab
packageDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(packageDir, 'data');
outDir = fullfile(packageDir, 'outputs');
```

## 6. Required input data

| File | Sheet | Purpose |
|---|---|---|
| `data/ResponseMatrix_1.xlsx` | first/default sheet | Measured response matrix for Fig. 3c |
| `data/ResponseMatrix_2.xlsx` | first/default sheet | Measured response matrix for Fig. 3h |
| `data/MeasuredSignals.xlsx` | `Fig3c` | Measured response vectors for the monochromatic spectra |
| `data/MeasuredSignals.xlsx` | `Fig3h` | Measured response vector for the dual-peak spectrum |
| `data/Fig3cSpectra.xlsx` | `ReferenceSpectra` | Wavelength axis and reference monochromatic spectra |
| `data/Fig3hSpectra.xlsx` | `ReferenceSpectra` | Wavelength axis and reference dual-peak spectrum |

The response-matrix files contain a wavelength column and sensor-response columns. The MATLAB script reads the numeric matrix, removes the first row and first column, normalizes the response kernel, and interpolates the wavelength-dependent data to 641 points.

## 7. Reconstruction parameters

| Parameter | Fig. 3c — monochromatic | Fig. 3h — dual-peak |
|---|---:|---:|
| Response matrix | `ResponseMatrix_1.xlsx` | `ResponseMatrix_2.xlsx` |
| Interpolated wavelength points | 641 | 641 |
| Gaussian basis functions | 641 | 641 |
| Gaussian-basis FWHM | 10.0 nm | 0.5 nm |
| Nonnegative solution | Yes | Yes |
| Integration rule | Basis integration | Basis integration |
| `alpha` | 0 | 0 |
| `beta` | `[0, 1e-5]` | `[0, 1e-4]` |
| Added noise | None | None |
| Output normalization | Divide by maximum | Divide by maximum |

Measured response vectors are loaded directly from `MeasuredSignals.xlsx` for each reconstruction.

## 8. How to run

In MATLAB, navigate to `1.monochromatic_dual-peak reconstruction` and run:

```matlab
Reconstruction_Main
```

Alternatively, from the repository root:

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

The CSV files contain wavelengths and normalized reconstructed spectra. The PNG files show the reconstructed spectra against the corresponding reference data. The MAT file stores the reconstruction and measurement structures.

## 10. Figure correspondence

| Figure | Reconstruction | Inputs | Outputs |
|---|---|---|---|
| Fig. 3c | Monochromatic | `ResponseMatrix_1.xlsx`, sheet `Fig3c`, `Fig3cSpectra.xlsx` | `Fig3c_Reconstruction.csv`, `.png` |
| Fig. 3h | Dual-peak | `ResponseMatrix_2.xlsx`, sheet `Fig3h`, `Fig3hSpectra.xlsx` | `Fig3h_Reconstruction.csv`, `.png` |

Both figures use `Reconstruction_Main.m` and the same `outputs/` directory.

---

# Part II. Broadband spectral reconstruction

## 11. Purpose

The Python workflow reconstructs two broadband spectra using measured photovoltaic response vectors and the device response matrix. The procedure comprises row normalization of the response matrix, Gaussian basis construction, generalized cross-validation (GCV) for the regularization parameter, and nonnegative least squares (NNLS).

Each dataset produces one measured-signal reconstruction, which is plotted with its reference spectrum.

## 12. Entry point and input data

```text
2.broadband reconstruction/main.py
```

The following processed files are read from the same directory as `main.py`:

| File | Contents | Dimensions / units |
|---|---|---|
| `ResponseMatrix.txt` | Measured photovoltaic response matrix | 201 response rows × 65 wavelength columns |
| `MeasuredSignals_1.txt` | Experimental drain-current response for Dataset 1 | 201 values, one column, **nA** |
| `MeasuredSignals_2.txt` | Experimental drain-current response for Dataset 2 | 201 values, one column, **nA** |
| `refspectra_1.txt` | Reference spectrum for Dataset 1 | Wavelength (nm), intensity |
| `refspectra_2.txt` | Reference spectrum for Dataset 2 | Wavelength (nm), intensity |

### Raw experimental data

The corresponding original measurement files are included under `Raw_data/`:

| Raw file | Corresponding processed data | Description |
|---|---|---|
| `Raw_data/measuredsignals_1.csv` | `MeasuredSignals_1.txt` | Original electrical measurement for Dataset 1, including drain current in amperes (A) |
| `Raw_data/measuredsignals_2.csv` | `MeasuredSignals_2.txt` | Original electrical measurement for Dataset 2, including drain current in amperes (A) |
| `Raw_data/reference_spectra_1.txt` | Reference-spectrum source data | Original spectrometer export containing wavelength (nm), intensity (counts), and acquisition metadata |

The processed `MeasuredSignals_1.txt` and `MeasuredSignals_2.txt` contain the measured `DRAIN-CURRENT` values expressed in nanoamperes:

```text
I_nA = I_A × 10^9
```

The two processed reference-spectrum files provide wavelength–intensity pairs for comparison with the reconstructed curves. `main.py` reads the processed TXT files; the raw files preserve the corresponding original measurement records.

## 13. Numerical configuration

| Parameter | Value |
|---|---|
| Wavelength range | 360–1000 nm |
| Wavelength samples | 65 |
| Gaussian basis functions | 65 |
| Response matrix | 201 × 65 |
| Response-matrix normalization | Each row divided by its row maximum |
| Measured-signal normalization | Divide by maximum before reconstruction |
| Regularization selection | Generalized cross-validation (GCV) |
| Regularization operator | Identity matrix |
| Constrained solver | SciPy `nnls` |
| Reconstructed-spectrum normalization | Divide by maximum |

The measured-signal Gaussian-basis widths are:

| Dataset | Measured-signal FWHM |
|---|---:|
| Dataset 1 | 60 nm |
| Dataset 2 | 120 nm |

For each dataset, `main.py` loads a single column of 201 measured current values, reconstructs one 65-point spectrum, and plots it against the corresponding two-column reference spectrum. The reference curve is normalized for plotting; it is not used as a second reconstruction input.

## 14. How to run

From the repository root:

```bash
python "2.broadband reconstruction/main.py"
```

Alternatively, from the broadband reconstruction directory:

```bash
cd "2.broadband reconstruction"
python main.py
```

The script uses `os.path.dirname(os.path.abspath(__file__))` to locate the input and output files, so execution does not depend on the current working directory.

## 15. Generated outputs

Each run displays two figures comparing the reconstructed broadband spectra with the reference spectra and saves:

```text
2.broadband reconstruction/
├── recon_1_YYYYMMDD_HHMMSS.txt
└── recon_2_YYYYMMDD_HHMMSS.txt
```

Each file contains **two columns** on the 360–1000 nm reconstruction grid:

```text
Wavelength_nm Reconstructed
```

The reconstructed intensity is normalized to a maximum of 1. The figures are displayed interactively and are not saved as image files by the current script.

---

# Part III. Houston2018 hyperspectral image classification

## 16. Purpose and entry point

```text
3. Image classification/Train.py
```

A run trains **three independent ten-class HybridViT models** using the same Houston2018 spatial train/validation/test split. Modes run consecutively in the order 1, 2, 3.

| Mode | Input representation | Channels | Result directory |
|---:|---|---:|---|
| 1 | Gaussian-fused RGB proxy derived from HSI | 3 | `mode_1_rgb/` |
| 2 | Twelve selected original HSI bands | 12 | `mode_2_native/` |
| 3 | RGB proxy with a simulated protanopia transformation | 3 | `mode_3_protanopia/` |

## 17. Input data and fixed classes

Place the Houston2018 files under `3. Image classification/data/Houston2018/`:

```text
data/Houston2018/
├── houston_hsi.mat
└── houston_gt.mat
```

The script loads the MATLAB variables `houston_hsi` (50 HSI bands) and `houston_gt` (ground-truth labels). The selected ten original class IDs are:

```python
CLASS_IDS = [1, 2, 3, 4, 8, 9, 11, 14, 15, 16]
```

These map, in order, to model output indices `0–9`. Each target is the label of the spatial patch's center pixel; exported predictions and confusion matrices use the original class IDs.

The ten-class set incorporates class 11 in place of class 7 following review of an earlier Mode-2 test run; the replacement was based on train/validation tile coverage. Results on this existing test split are exploratory.

The script validates fixed SHA-256 fingerprints:

| Object | Expected SHA-256 |
|---|---|
| Ground truth (filtered `uint8`) | `3d287d75afc131717b634179ae4476e7d0c8cd3e1149ee8282a80a03f4ce5910` |
| Raw HSI array | `84588a62ca5feb936ed45e17566be9721ce2620bef3585ec299bb4b4aa5dcb4d` |
| Shared spatial split | `e91e11f08255d957b57ff92a91ae2b3fea60c9ced2bf7b9e40d9173f96bbe46f` |

## 18. Spatial train/validation/test split

A shared spatial split is constructed across all 16 original classes before filtering to the selected ten. Eligible sample centers are assigned to **64 × 64-pixel tiles** and partitioned using SciPy mixed-integer linear programming (`milp`). Training, validation, and test tiles are disjoint; each eligible sample neighborhood remains within its tile.

| Parameter | Value |
|---|---:|
| Spatial tile size | 64 × 64 |
| Eligibility window (`IMG_SIZE`) | 32 × 32 |
| Extracted spatial patch | 31 × 31 |
| Training sample cap | 1,000 per original class |
| Validation sample cap | 250 per original class |
| Test sample cap | 40,000 total |
| Minimum reserved test samples per original class, if capped | 10 |
| Minimum per-class region count | 10 |
| MILP time limit | 60 s |
| Random seed | 42 |

The common split is stored at:

```text
results_houston_tensorflow/spatial_16classes_seed42_img32_tile64/shared_split_coords.npz
```

`Train.py` reads a saved split or generates one and checks its coordinate digest. A matching existing split can also be specified using `--split-file`.

## 19. Input preprocessing

### Mode 1: RGB proxy

Fifty original HSI bands are combined using normalized Gaussian spectral weights:

| Channel | Gaussian center | Gaussian σ |
|---|---:|---:|
| Red | 650 nm | 55 nm |
| Green | 550 nm | 45 nm |
| Blue | 460 nm | 40 nm |

An optional `--wavelengths` file supplies 50 strictly increasing wavelength values. Otherwise, the script uses a uniformly spaced 380–1000 nm wavelength axis.

### Mode 2: Native HSI bands

Mode 2 selects 12 native HSI channels directly from the original HSI cube, using zero-based indices:

```python
MODE2_BAND_INDICES = [0, 4, 9, 13, 18, 22, 27, 31, 36, 40, 45, 49]
```

This mode uses the original measured bands without PCA or RGB-to-spectrum conversion.

### Mode 3: Simulated protanopia

Mode 3 transforms the RGB proxy with the matrix below and clips values to `[0, 1]`:

```text
[[ 0.152286,  1.052583, -0.204868],
 [ 0.114503,  0.786281,  0.099216],
 [-0.003882, -0.048116,  1.051998]]
```

### Common intensity scaling

An intensity scale is estimated from the **99.5th percentile of original 16-class training-center spectra**. All three representations use that training-derived scale and are clipped to `[0, 1]`. Validation and test samples do not contribute to its estimation. Each mode exports `preprocessing.json`.

## 20. Samples and augmentation

Each image-level sample contains a `31 × 31 × C` spatial patch and a center-pixel spectral vector of length `C`, with `C = 3` or `12`. The center-pixel label defines the class target.

Only training samples receive random augmentation:

| Operation | Configuration |
|---|---|
| Horizontal flip | Probability 0.5 |
| Vertical flip | Probability 0.5 |
| Rotation in 90° increments | Probability 0.75 |
| Multiplicative intensity gain | Probability 0.8; uniform 0.75–1.25 |
| Gaussian noise | Probability 0.5; σ = 0.01 |
| Batch MixUp | Probability 0.2; Beta(0.2, 0.2) mixing weights |

Training samples are shuffled each epoch. Validation and test samples are evaluated without augmentation.

## 21. HybridViT architecture

All three modes use the same two-branch **HybridViT** architecture:

1. A three-layer convolutional stem converts the spatial patch into an `8 × 8` feature map (64 spatial tokens).
2. A learned CLS token and positional embeddings form 65 Transformer tokens.
3. Three Transformer encoder blocks extract spatial context.
4. A separate 1D-convolutional branch processes the center-pixel spectral vector.
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

The network produces logits. Training uses smoothed categorical cross-entropy with `from_logits=True`.

## 22. Training configuration

Each mode is trained independently for **50 epochs**, without early stopping.

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
| Checkpoint criterion | Validation macro-F1 evaluated with EMA weights |

At each epoch end, EMA-weight validation and test metrics are logged. The checkpoint is selected by **validation macro-F1**. The selected model is then evaluated on unaugmented training, validation, and test samples. Outputs include overall accuracy, macro-F1, Cohen's kappa, and per-class precision, recall, F1, and support.

## 23. Run instructions

From the repository root:

```bash
cd "3. Image classification"
python Train.py
```

Windows PowerShell:

```powershell
cd ".\3. Image classification"
python .\Train.py
```

To display the configuration without training:

```bash
python Train.py --dry-run
```

Example with explicit options:

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
| `--data-dir` | `data/Houston2018/` | Houston HSI and ground-truth MAT files |
| `--output` | `results_houston_tensorflow/` | Output directory |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--batch-size` | `32` | Batch size |
| `--wavelengths` | None | Optional 50-element wavelength vector for Modes 1 and 3 |
| `--split-file` | None | Existing matching spatial split |
| `--tag` | Timestamped name | Run identifier |
| `--dry-run` | Off | Display experiment plan |

A standard run executes all three modes sequentially. Different tags separate the outputs of independent runs.

## 24. Output files

The script creates the following structure under `3. Image classification/`:

```text
results_houston_tensorflow/
└── spatial_16classes_seed42_img32_tile64/
    ├── shared_split_coords.npz            # generated if absent
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

Each `mode_*` directory contains:

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

`best_model.keras` and `model.keras` contain the selected EMA checkpoint; `last_model.keras` contains the last-epoch raw model. The prediction archives include pixel coordinates and original class IDs. Confusion matrices are exported as counts and row-normalized percentages. `training_comparison.csv` summarizes performance across the three modes.

---

# Part IV. Figure reproduction map

## 25. Fig. 3c — monochromatic reconstruction

**Entry point**

```text
1.monochromatic_dual-peak reconstruction/Reconstruction_Main.m
```

**Inputs**

```text
1.monochromatic_dual-peak reconstruction/data/ResponseMatrix_1.xlsx
1.monochromatic_dual-peak reconstruction/data/MeasuredSignals.xlsx   # sheet: Fig3c
1.monochromatic_dual-peak reconstruction/data/Fig3cSpectra.xlsx      # sheet: ReferenceSpectra
```

**Parameters**

```text
641 interpolated wavelength samples
641 Gaussian basis functions
Gaussian FWHM = 10.0 nm
Nonnegative quadratic optimization
alpha = 0; beta = [0, 1e-5]
Maximum-intensity normalization
```

**Outputs**

```text
1.monochromatic_dual-peak reconstruction/outputs/Fig3c_Reconstruction.csv
1.monochromatic_dual-peak reconstruction/outputs/Fig3c_Reconstruction.png
```

## 26. Fig. 3g — spectral reconstruction

**Entry point**

```text
2.broadband reconstruction/main.py
```

**Inputs**

```text
2.broadband reconstruction/ResponseMatrix.txt
2.broadband reconstruction/MeasuredSignals_1.txt
2.broadband reconstruction/MeasuredSignals_2.txt
2.broadband reconstruction/refspectra_1.txt
2.broadband reconstruction/refspectra_2.txt
```

Original electrical measurements are included as `Raw_data/measuredsignals_1.csv` and `Raw_data/measuredsignals_2.csv` (drain current in A); the processed measured signals are in **nA**. `Raw_data/reference_spectra_1.txt` preserves an original spectrometer reference export.

**Parameters**

```text
Response matrix = 201 × 65
Wavelength range = 360–1000 nm; 65 points
Gaussian basis functions = 65
Dataset 1 measured-signal FWHM = 60 nm
Dataset 2 measured-signal FWHM = 120 nm
Row-normalized response matrix
GCV regularization and nonnegative least squares (NNLS)
Maximum-intensity normalization
```

**Outputs**

```text
2.broadband reconstruction/recon_1_YYYYMMDD_HHMMSS.txt
2.broadband reconstruction/recon_2_YYYYMMDD_HHMMSS.txt
```

Each output contains `Wavelength_nm Reconstructed`; two reference-comparison figures are displayed.

## 27. Fig. 3h — dual-peak reconstruction

**Entry point**

```text
1.monochromatic_dual-peak reconstruction/Reconstruction_Main.m
```

**Inputs**

```text
1.monochromatic_dual-peak reconstruction/data/ResponseMatrix_2.xlsx
1.monochromatic_dual-peak reconstruction/data/MeasuredSignals.xlsx   # sheet: Fig3h
1.monochromatic_dual-peak reconstruction/data/Fig3hSpectra.xlsx      # sheet: ReferenceSpectra
```

**Parameters**

```text
641 interpolated wavelength samples
641 Gaussian basis functions
Gaussian FWHM = 0.5 nm
Nonnegative quadratic optimization
alpha = 0; beta = [0, 1e-4]
Maximum-intensity normalization
```

**Outputs**

```text
1.monochromatic_dual-peak reconstruction/outputs/Fig3h_Reconstruction.csv
1.monochromatic_dual-peak reconstruction/outputs/Fig3h_Reconstruction.png
```

The MATLAB run also saves the shared `outputs/ReproductionResults_vdwRS2_loaded.mat` file for Fig. 3c and Fig. 3h.

## 28. Houston2018 three-mode classification comparison

**Entry point**

```text
3. Image classification/Train.py
```

**Inputs**

```text
3. Image classification/data/Houston2018/houston_hsi.mat
3. Image classification/data/Houston2018/houston_gt.mat
```

An existing matching spatial split can be supplied using `--split-file`. A 50-element wavelength file may be supplied using `--wavelengths` for RGB-proxy modes.

**Parameters**

```text
Selected classes = [1, 2, 3, 4, 8, 9, 11, 14, 15, 16]
Shared spatial train/validation/test split; 64 × 64 tiles
Center-pixel classification using 31 × 31 spatial patches
Mode 1 = 3-channel Gaussian RGB proxy
Mode 2 = 12 selected native HSI bands
Mode 3 = 3-channel simulated protanopia proxy
Centered spatial–spectral HybridViT
50 epochs per mode; batch size 32; random seed 42
EMA decay = 0.99; checkpoint selection by validation macro-F1
```

**Outputs**

```text
3. Image classification/results_houston_tensorflow/
└── spatial_16classes_seed42_img32_tile64/
    └── <tag>/
        ├── training_comparison.csv
        ├── mode_accuracy_comparison.png
        ├── mode_macro_f1_comparison.png
        ├── summary.json
        ├── mode_1_rgb/                # model, logs, predictions, metrics, confusion matrices
        ├── mode_2_native/             # model, logs, predictions, metrics, confusion matrices
        └── mode_3_protanopia/         # model, logs, predictions, metrics, confusion matrices
```

---

## 29. Citation

If this repository is used in academic work, please cite the corresponding manuscript.

```text
Citation information will be added after publication.
```

## 30. Contact and code-use policy

This repository is provided for academic reproducibility and methodological review. For questions about the data, code, permissions, or collaboration, contact the corresponding author. Use and redistribution should follow the license or permission terms supplied with the repository.
