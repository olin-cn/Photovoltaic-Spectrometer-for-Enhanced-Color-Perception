# Photovoltaic Spectrometer for Enhanced Color Perception

This repository provides the code and example data used to reproduce the spectral-reconstruction and image-classification workflows associated with our vdWH photovoltaic spectrometer study. The repository is organized as three independent workflows:

1. monochromatic and dual-peak spectral reconstruction;
2. broadband spectral reconstruction; and
3. CIFAR-10 image classification using RGB, simulated multispectral, or simulated protanopia inputs.

The purpose of this repository is to make the data flow, numerical parameters, file dependencies, and output-generation procedures explicit and executable.

---

## 1. Important scope clarification

The spectral-reconstruction workflows use experimentally measured response matrices and measured photovoltaic response vectors supplied with the repository.

The image-classification workflow does **not** perform direct physical imaging with the single-pixel spectrometer. In the current public `Train.py`, the `MODE = 2` branch generates a **simulated wavelength-resolved representation** from CIFAR-10 RGB images using predefined Gaussian spectral basis functions, simulated photon/read noise, PCA, and standardization. The current script does not directly load the response-matrix files used by the spectral-reconstruction workflows.

Accordingly, the `MODE = 2` branch should be described as a simulated or spectrometer-inspired multispectral proof-of-concept unless the classification code is further revised to load and use experimentally measured device-response data.

---

## 2. Repository structure

The expected repository layout is:

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
│   └── outputs/
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
│   ├── cifar-10-python.tar.gz
│   ├── results/
│   └── spectral_cache/          # created automatically for MODE = 2
│
└── README.md
```

### File-name consistency

The current MATLAB entry script reads:

```text
data/ResponseMatrix_1.xlsx
data/ResponseMatrix_2.xlsx
```

The repository must therefore provide these exact file names. References to `RMatrix_1.xlsx`, `RMatrix_2.xlsx`, or a single `ResponseMatrix.xlsx` in older documentation should be removed unless the corresponding scripts are also changed.

File names are case-sensitive on Linux and some server environments. In the broadband workflow, the code and repository should use the same capitalization for:

```text
Refspectra_1.txt
Refspectra_2.txt
```

---

## 3. Tested software environment

The workflows require the following software.

### MATLAB workflow

- MATLAB R2020b or later is recommended.
- No Optimization Toolbox is required by the current `Reconstruction_Main.m`, because it contains a local active-set nonnegative quadratic-programming solver.
- Earlier MATLAB releases may also work if they support the functions used by the script, including `readmatrix`, `readtable`, `interp1`, `writetable`, and local functions in scripts.

### Python workflows

Recommended:

```text
Python 3.9 or later
TensorFlow 2.18.0
NumPy
SciPy
Pandas
Matplotlib
scikit-learn
tqdm
```

Install the Python dependencies with:

```bash
pip install numpy scipy pandas matplotlib scikit-learn tqdm tensorflow
```

A CUDA-enabled TensorFlow installation can accelerate classification training. A GPU is not required: `Train.py` detects available GPUs and otherwise continues on the CPU.

For strict reproduction, record the Python, TensorFlow, CUDA, cuDNN, operating-system, and GPU versions used for each run.

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
| `data/MeasuredSignals.xlsx` | `Fig3c` | measured response matrix for the Fig. 3c spectra |
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

To make execution independent of the current working directory, a future code revision should resolve all input paths from `Path(__file__).resolve().parent`, as already done in the MATLAB and classification workflows.

---

# Part III. CIFAR-10 image classification

## 16. Purpose and entry point

The classification workflow compares three image representations using the same Vision Transformer backbone:

```text
3. Image classification/Train.py
```

The three supported modes are:

| `CONFIG.MODE` | Input representation | Network input channels |
|---:|---|---:|
| `1` | normalized RGB with horizontal-flip augmentation and Gaussian noise | 3 |
| `2` | simulated wavelength-resolved representation followed by PCA | 12 |
| `3` | simulated protanopia transformation | 3 |

### Correct mode for the multispectral branch

To execute the simulated multispectral branch used for the intended Fig. 5 multispectral comparison, set:

```python
class CONFIG:
    MODE = 2
```

The currently supplied script has `MODE = 3` as its checked-in default. This value executes the simulated protanopia branch and therefore should not be described as the multispectral Fig. 5 configuration. Before public release, either:

1. change the default to `MODE = 2`; or
2. retain `MODE = 3` but state explicitly that users must set `MODE = 2` to reproduce the multispectral branch.

## 17. Dataset preparation

The workflow uses the CIFAR-10 Python archive:

```text
cifar-10-python.tar.gz
```

Place it in the same folder as `Train.py`:

```text
3. Image classification/
├── Train.py
└── cifar-10-python.tar.gz
```

The script loads the archive directly and does not extract it permanently.

The official archive contains:

- 50,000 training images;
- 10,000 test images;
- 10 classes;
- image size `32 × 32 × 3`.

## 18. Fixed random seed

The current script defines:

```python
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

For stricter deterministic execution, the released code should also use:

```python
import random

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
```

Even with deterministic settings, small numerical differences may remain across TensorFlow, CUDA, cuDNN, operating-system, and GPU versions.

## 19. Input preprocessing

### 19.1 Common preprocessing

All CIFAR-10 pixel values are converted from integer values in `[0, 255]` to `float32` values in `[0, 1]`.

Training augmentation uses a horizontal flip with probability 0.5.

### 19.2 `MODE = 1`: RGB input

The training RGB images are horizontally augmented and Gaussian noise with standard deviation `0.03` is added. Values are clipped to `[0, 1]`.

### 19.3 `MODE = 2`: simulated wavelength-resolved input

The current script performs the following operations:

1. add Gaussian image noise with standard deviation `0.03`;
2. create 24 wavelength samples uniformly spanning 400–1000 nm;
3. construct four predefined Gaussian-like spectral curves centered near 450, 550, 650, and 850 nm;
4. concatenate the RGB channels with the per-pixel RGB mean to create four coefficients;
5. linearly combine the four coefficients with the spectral curves;
6. add simulated photon noise with scale `0.005 × sqrt(abs(signal))`;
7. add simulated read noise with standard deviation `0.002`;
8. amplify spectral-band indices `[0, 5, 12, 18]` by a factor `1.3 + N(0, 0.1)`;
9. flatten the spectral pixels;
10. reduce the 24 bands to 12 PCA components;
11. standardize the PCA components; and
12. reshape the data to `32 × 32 × 12`.

This branch is generated from CIFAR-10 RGB values and predefined spectral curves. It does not currently read `ResponseMatrix_1.xlsx`, `ResponseMatrix_2.xlsx`, or `ResponseMatrix.txt`.

### 19.4 `MODE = 3`: simulated protanopia input

The RGB values are transformed using:

```text
[[ 0.152286,  1.052583, -0.204868],
 [ 0.114503,  0.786281,  0.099216],
 [-0.003882, -0.048116,  1.051998]]
```

The transformed images are scaled, perturbed with Gaussian noise, clipped to `[0, 1]`, and returned as three-channel images.

## 20. Vision Transformer configuration

The current model configuration is:

| Parameter | Value |
|---|---:|
| Image size | 32 × 32 |
| Patch size | 4 |
| Projection dimension | 192 |
| Transformer layers | 6 |
| Attention heads | 8 |
| MLP ratio | 4 |
| Attention/MLP dropout | 0.3 |
| Final pooling | global average pooling |
| Dense hidden layer | 256 units, `swish` |
| Number of classes | 10 |
| Output activation | softmax |

The model applies:

1. a `3 × 3` convolution with 64 filters;
2. optional batch normalization for modes 2 and 3;
3. a strided patch-projection convolution;
4. trainable positional embeddings;
5. six Transformer encoder blocks;
6. global average pooling;
7. dropout;
8. a 256-unit dense layer; and
9. a 10-class softmax output.

## 21. Training configuration

The checked-in configuration is:

| Parameter | Value |
|---|---:|
| Epochs | 100 |
| Batch size | 128 |
| Learning rate | `3e-4` |
| Optimizer | AdamW when available; otherwise Adam |
| Weight decay | `1e-3` |
| Label smoothing | 0.2 |
| Loss | categorical cross-entropy |
| Training shuffle buffer | 10,000 |
| Random seed | 42 |
| Output folder | `results/` |
| Cache folder | `spectral_cache/` |

No early-stopping callback is used in the current script.

## 22. Current model-selection and evaluation behavior

The current public `Train.py`:

- loads all 50,000 CIFAR-10 training images;
- discards the official 10,000-image test split in the statement  
  `(x_train, y_train), _ = load_cifar10_from_tar_gz(...)`;
- does not create a validation split;
- monitors training `accuracy` when saving `best_model.keras`;
- reloads the checkpoint with the highest training accuracy;
- computes predictions and a confusion matrix on the training set; and
- does not calculate an independent validation or test accuracy.

Therefore, the current script is sufficient to demonstrate the computational branch and model training, but it is **not sufficient by itself to substantiate an independently evaluated Fig. 5 accuracy value**.

This limitation must not be hidden in the README.

## 23. Dataset usage, model selection, and image-level accuracy

The accuracy values reported in Fig. 5 are **image-level training accuracies** calculated using all 50,000 images in the official CIFAR-10 training split. No separate validation subset was used, and the official CIFAR-10 test split was not used to calculate the reported Fig. 5 values.

The random seed was fixed at `42`, and each model was trained for `100` epochs. The checkpoint with the highest training accuracy was selected using:

```python
tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="accuracy",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
```

Each CIFAR-10 image was treated as one image-level classification sample. The ViT patches were used only as internal feature tokens and were not treated as independent samples.

For each image, the predicted label was defined as the class with the highest softmax probability:

```python
predicted_labels = np.argmax(
    model.predict(train_ds_eval, verbose=1),
    axis=1
)
```

The reported image-level training accuracy was calculated as:

```python
image_level_training_accuracy = np.mean(
    predicted_labels[:len(y_train_true)] == y_train_true
)
```

Equivalently,

```text
image-level training accuracy
    = number of correctly classified training images
      / 50,000
```
## 24. Spectral cache

For `MODE = 2`, the current script writes:

```text
spectral_cache/spectral_train.npy
```

The cache file name does not encode the spectral parameters, PCA dimension, random seed, or data split. Delete the cache whenever any preprocessing parameter or split changes.

For a robust release, use configuration-specific cache names, for example:

```text
spectral_train_mode2_seed42_bands24_pca12.npy
spectral_val_mode2_seed42_bands24_pca12.npy
spectral_test_mode2_seed42_bands24_pca12.npy
```

The release should also store the fitted PCA and standardization parameters.

## 25. How to run the current classification script

### Windows PowerShell

```powershell
cd ".\3. Image classification"
python .\Train.py
```

### Linux/macOS

```bash
cd "3. Image classification"
python Train.py
```

To run the multispectral branch, first set:

```python
CONFIG.MODE = 2
```

The script creates `results/` automatically. For `MODE = 2`, it also creates `spectral_cache/`.

## 26. Outputs generated by the current script

The current script writes:

```text
results/best_model.keras
results/model.keras
results/training_log.csv
results/history.csv
results/confusion_matrix_train_percentage.csv
results/training_curves.png
```

These are training-set outputs. They should not be relabeled as independent test-set results.

---

# Part IV. End-to-end figure reproduction map

## 27. Fig. 3c

```text
Input:
  data/ResponseMatrix_1.xlsx
  data/MeasuredSignals.xlsx, sheet Fig3c
  data/Fig3cSpectra.xlsx, sheet ReferenceSpectra

Entry point:
  1.monochromatic_dual-peak reconstruction/Reconstruction_Main.m

Key settings:
  641 wavelength points
  641 Gaussian basis functions
  FWHM = 10.0 nm
  second-order penalty = 1e-5
  nonnegative reconstruction

Outputs:
  outputs/Fig3c_Reconstruction.csv
  outputs/Fig3c_Reconstruction.png
```

## 28. Fig. 3h (or Fig. 3g after verified renumbering)

```text
Input:
  data/ResponseMatrix_2.xlsx
  data/MeasuredSignals.xlsx, sheet Fig3h
  data/Fig3hSpectra.xlsx, sheet ReferenceSpectra

Entry point:
  1.monochromatic_dual-peak reconstruction/Reconstruction_Main.m

Key settings:
  641 wavelength points
  641 Gaussian basis functions
  FWHM = 0.5 nm
  second-order penalty = 1e-4
  nonnegative reconstruction

Outputs:
  outputs/Fig3h_Reconstruction.csv
  outputs/Fig3h_Reconstruction.png
```

## 29. Fig. 5

### Current executable branch

```text
Input:
  CIFAR-10 Python archive

Entry point:
  3. Image classification/Train.py

Multispectral setting:
  CONFIG.MODE = 2

Current processing:
  RGB normalization
  horizontal flip
  synthetic 24-band spectral conversion over 400–1000 nm
  simulated photon/read noise
  PCA to 12 components
  standardization
  ViT training for 100 epochs
```

### Requirement for reported image-level accuracy

The exact Fig. 5 result must additionally document and implement:

```text
fixed train/validation/test split
split seed and split indices
training-only PCA/scaler fitting
validation-based checkpoint selection
official test-set evaluation
image-level accuracy calculation
saved predictions and configuration
```

Until those steps are present in the released code, the repository should describe the current Fig. 5 script as a proof-of-concept training workflow rather than a complete independent reproduction of the reported accuracy.

---

## 30. Reproducibility checklist

Before creating a release, verify all of the following:

- [ ] All file names in the scripts match the repository exactly.
- [ ] `ResponseMatrix_1.xlsx` and `ResponseMatrix_2.xlsx` are present in `data/`.
- [ ] The README directory tree matches the actual repository.
- [ ] The Fig. 3g/Fig. 3h panel number is consistent everywhere.
- [ ] The broadband reference-spectrum capitalization is consistent.
- [ ] `CONFIG.MODE = 2` is used for the multispectral Fig. 5 branch.
- [ ] The classification data split is fixed and saved.
- [ ] PCA and standardization are fitted only on training data.
- [ ] Best-model selection uses `val_accuracy`.
- [ ] Test accuracy is calculated only after model selection.
- [ ] The image-level accuracy procedure is included in the release.
- [ ] Software versions and hardware are recorded.
- [ ] Generated outputs can be reproduced from a clean checkout.

---

## 31. Known limitations

- The MATLAB reconstruction uses a Gaussian dictionary and regularized nonnegative optimization; performance may decrease for high-noise or highly complex spectra.
- The broadband examples use supplied measured response vectors rather than a live hardware-acquisition pipeline.
- The current classification `MODE = 2` branch is simulated from CIFAR-10 RGB images and predefined spectral curves rather than direct multispectral imaging.
- The current checked-in classification script does not yet implement validation/test evaluation.
- Runtime, memory use, and model efficiency have not been optimized for embedded deployment.

---

## 32. Citation

If this repository is used in academic work, please cite the corresponding manuscript.

```text
Citation information will be added after publication.
```

---

## 33. Contact and code-use policy

This repository is provided for academic reproducibility and methodological review.

For questions about the data, code, permissions, or collaboration, contact the corresponding author. Any copying, redistribution, commercial use, or derivative use should follow the license or written permission terms supplied with the final repository release.
