# Fig. 3c / Fig. 3h Reproduction Package

This folder is a reviewer-facing reproduction package based on the original `vdwRS2.m` workflow.

## Main entry points

1. `run_vdwRS2_loaded_Fig3c_Fig3h.m`
   - Strict version based on `vdwRS2.m`.
   - The code before `%% Functions` only loads response matrices, measured signals, and spectra from Excel files, then calls the reconstruction functions.
   - The reconstruction function block is based on `vdwRS2.m`; unused spectrum-construction helper functions were removed from the delivered script.
   - This version keeps the original `quadprog` call and therefore requires MATLAB Optimization Toolbox.

2. `run_vdwRS2_loaded_Fig3c_Fig3h_no_optim_toolbox.m`
   - Dependency-free reviewer version.
   - It uses the same loading and reconstruction workflow, but replaces the original `quadprog` call in `SolveLinearForm` with a local active-set nonnegative quadratic-programming solver.
   - Use this script if MATLAB reports that Optimization Toolbox must be installed.

## Why Optimization Toolbox was requested

The original `vdwRS2.m` solves the linear inverse reconstruction through MATLAB functions `optimoptions` and `quadprog`. Both functions are provided by Optimization Toolbox. Therefore, running the strict original-code version on a MATLAB installation without that Add-On will produce a missing-toolbox error.

To avoid requiring reviewers to install an Add-On, the `*_no_optim_toolbox.m` script keeps the same data loading and reconstruction pipeline but replaces only the quadratic-programming backend with a local solver.

## Loaded data

All measured signals and spectra are loaded from Excel files in `data/`. The main scripts do not contain any spectrum-generation or inverse-design process for these measured signals.

- `data/RMatrix_1.xlsx`
  - Response matrix used for Fig. 3c.

- `data/RMatrix_2.xlsx`
  - Response matrix used for Fig. 3h.

The response matrices are interpolated to 641 wavelength rows before reconstruction.

- `data/MeasuredSignals.xlsx`
  - Sheet `Fig3c`: measured signal matrix for Fig. 3c, size `6 x 201`.
  - Sheet `Fig3h`: measured signal vector for Fig. 3h, size `1 x 201`.

- `data/Fig3cSpectra.xlsx`
  - Sheet `ReferenceSpectra`: wavelength axis and reference spectra used for plotting.

- `data/Fig3hSpectra.xlsx`
  - Sheet `ReferenceSpectra`: wavelength axis and the measured single-peak-sum target spectrum used for Fig. 3h plotting.
  - The Fig. 3h target spectrum is loaded directly from column 2 of `ReferenceSpectra`; the individual measured single peaks are not stored or plotted in this reproduction package.
  - `MeasuredSignals2` is unchanged and is still loaded from `data/MeasuredSignals.xlsx`.

The reconstruction basis is not loaded from any spectrum sheet. The scripts use a generic Gaussian dictionary with fixed FWHM and regularization settings. For Fig. 3c, the Gaussian basis FWHM is set to 10.0 nm. No noise-level field or noise injection is used in the reproduction scripts.

## Linear relation between RMatrix and MeasuredSignals

The measured signals are linearly related to the response matrix. For each spectrum, each sensor channel is the wavelength integral of:

```text
MeasuredSignal(sensor) = integral( Spectrum(wavelength) * RMatrix(wavelength, sensor) d(wavelength) )
```

In the scripts this relation is used in the inverse direction: `MeasuredSignals1` and `MeasuredSignals2` are loaded as fixed measured responses, and `vdwRS2` reconstructs spectra from those responses through the response matrix.

## How to run

In MATLAB, set this folder as the current folder and run one of:

```matlab
run_vdwRS2_loaded_Fig3c_Fig3h
```

or, without Optimization Toolbox:

```matlab
run_vdwRS2_loaded_Fig3c_Fig3h_no_optim_toolbox
```

## Outputs

The scripts write results to `outputs/`:

- `Fig3c_reproduced_vdwRS2_loaded.png`
- `Fig3h_reproduced_vdwRS2_loaded.png`
- `Fig3h_reconstructed_curve.csv`
- `Fig3c_metrics.csv`
- `Fig3h_metrics.csv`
- `ReproductionResults_vdwRS2_loaded.mat`

Recent verification with the strict original-`quadprog` script gave the following response-space errors:

```text
Fig. 3c max RMSE = 8.989752e-04, max abs error = 3.071043e-03
Fig. 3h RMSE = 5.772533e-05, max abs error = 1.636089e-04
```

Recent verification with the no-toolbox script gave:

```text
Fig. 3c max RMSE = 1.264012e-03, max abs error = 1.402787e-02
Fig. 3h RMSE = 1.184228e-06, max abs error = 3.738675e-06
```

After replacing the Fig. 3h plotting target with the measured single-peak sum, the no-toolbox script was rerun. The loaded `MeasuredSignals2` vector was not modified; only the displayed Fig. 3h target spectrum was replaced. The Fig. 3h output shows the reconstructed spectrum as a blue solid line and the normalized measured-sum target as a gray dashed line.
