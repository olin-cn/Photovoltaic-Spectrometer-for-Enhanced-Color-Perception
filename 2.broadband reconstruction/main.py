import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.linalg import svd
from scipy.optimize import nnls, fmin_slsqp

# ================== settings ==================
NumWavelengths = 65
NumGaussianBasis = 65
MinLambda = 360
MaxLambda = 1000

ResponseMatrixFile = 'ResponseMatrix.txt'
MeasuredFiles = {
    '1': 'MeasuredSignals_1.txt',
    '2': 'MeasuredSignals_2.txt',
}
ReferenceFiles = {
    '1': 'refspectra_1.txt',
    '2': 'refspectra_2.txt',
}

# FWHM for measured-signal reconstruction only (nm)
FWHM_map = {'1': 60, '2': 120}

script_folder = os.path.dirname(os.path.abspath(__file__))

# ================== load response ==================
ResponseMatrix = np.loadtxt(os.path.join(script_folder, ResponseMatrixFile))
if ResponseMatrix.ndim != 2 or ResponseMatrix.shape[1] != NumWavelengths:
    raise ValueError(
        f'ResponseMatrix must have {NumWavelengths} wavelength columns; '
        f'got shape {ResponseMatrix.shape}'
    )
row_max = np.max(ResponseMatrix, axis=1, keepdims=True)
ResponseMatrix = ResponseMatrix / np.where(row_max != 0, row_max, 1.0)

VecOfLambdas = np.linspace(MinLambda, MaxLambda, NumWavelengths) / MaxLambda
VecOfLambdasPlot_nm = VecOfLambdas * MaxLambda
NumRow = ResponseMatrix.shape[0]

ResponseMatrixInterp = np.zeros((NumRow, NumWavelengths))
for r in range(NumRow):
    spline = interpolate.interp1d(
        VecOfLambdasPlot_nm,
        ResponseMatrix[r, :],
        kind='cubic',
        bounds_error=False,
        fill_value=0.0,
    )
    ResponseMatrixInterp[r, :] = spline(VecOfLambdasPlot_nm)

# ================== gaussian basis ==================
def build_gaussian(FWHM):
    centers = np.linspace(MinLambda, MaxLambda, NumGaussianBasis) / MaxLambda
    sigma = FWHM / (MaxLambda * (2 * np.sqrt(2 * np.log(2))))
    G = np.zeros((NumWavelengths, NumGaussianBasis))
    for j in range(NumGaussianBasis):
        G[:, j] = np.exp(-0.5 * ((VecOfLambdas - centers[j]) / sigma) ** 2)
    return G


# ================== GCV ==================
def find_gamma(c, A):
    U, s, _ = svd(A, full_matrices=False)
    m = A.shape[0]
    s = np.maximum(s, 1e-12)

    def gcv(logg):
        g = np.exp(logg)
        fi = s**2 / (s**2 + g**2)
        Utc = U.T @ c
        num = np.sum(((1 - fi) * Utc) ** 2)
        den = (m - np.sum(fi)) ** 2
        return 1e100 if den < 1e-12 else num / den

    try:
        res = fmin_slsqp(lambda x: gcv(x[0]), x0=[-20], bounds=[(-50, 50)], disp=False)
        return float(np.exp(res[0]))
    except Exception:
        return 1e-6


# ================== reconstruct measured signal ==================
def reconstruct(measured_signal, FWHM):
    y = np.asarray(measured_signal, dtype=float).reshape(-1)
    if y.size != NumRow:
        raise ValueError(
            f'Measured signal must contain {NumRow} values (one per response row); '
            f'got {y.size}'
        )

    G = build_gaussian(FWHM)
    W = ResponseMatrixInterp @ G
    if np.max(W) > 0:
        W = W / np.max(W)
    L = np.eye(NumGaussianBasis)

    if np.max(y) > 0:
        y = y / np.max(y)

    gamma = find_gamma(y, W)
    A = np.vstack([W, np.sqrt(gamma) * L])
    b = np.concatenate([y, np.zeros(L.shape[0])])
    c, _ = nnls(A, b)

    result = G @ c
    if np.max(result) > 0:
        result = result / np.max(result)
    return result


# ================== plot ==================
def plot_reconstruction(result, reference_data, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        VecOfLambdasPlot_nm,
        result,
        color='r',
        linewidth=1.8,
        label='Reconstructed',
    )

    # Keep reference comparison, but never reconstruct an ideal curve.
    ref_curve = reference_data[:, 1].copy()
    if np.max(ref_curve) > 0:
        ref_curve /= np.max(ref_curve)
    ax.plot(
        reference_data[:, 0],
        ref_curve,
        '--k',
        linewidth=1.2,
        alpha=0.7,
        label='Reference',
    )

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


# ================== run ==================
def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for d in ['1', '2']:
        # MeasuredSignals_X.txt contains ONE column only.
        measured_signal = np.loadtxt(os.path.join(script_folder, MeasuredFiles[d]))
        if measured_signal.ndim != 1:
            raise ValueError(f'{MeasuredFiles[d]} must contain exactly one column.')

        reference_data = np.loadtxt(os.path.join(script_folder, ReferenceFiles[d]))
        if reference_data.ndim != 2 or reference_data.shape[1] < 2:
            raise ValueError(f'{ReferenceFiles[d]} must contain wavelength and intensity columns.')

        result = reconstruct(measured_signal, FWHM_map[d])
        plot_reconstruction(result, reference_data, f'Reconstruction {d}')

        output_path = os.path.join(script_folder, f'recon_{d}_{timestamp}.txt')
        np.savetxt(
            output_path,
            np.column_stack((VecOfLambdasPlot_nm, result)),
            fmt='%.6f',
            header='Wavelength_nm Reconstructed',
        )
        print('Saved:', output_path)

    plt.show()


if __name__ == '__main__':
    main()
