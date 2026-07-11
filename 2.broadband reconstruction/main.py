import numpy as np
from scipy import interpolate
from scipy.linalg import svd
from scipy.optimize import nnls, fmin_slsqp
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ================== settings ==================
NumWavelengths = 65
NumGaussianBasis = 65

MinLambda = 360
MaxLambda = 1000

ResponseMatrixFile = 'ResponseMatrix.txt'

MeasuredFiles = {
    "1": "MeasuredSignals_1.txt",
    "2": "MeasuredSignals_2.txt"
}

ReferenceFiles = {
    "1": "refspectra_1.txt",
    "2": "refspectra_2.txt"
}

# ================== FWHM ==================
FWHM_map = {
    ("1", "ideal"): 10,
    ("1", "exp"): 60,
    ("2", "ideal"): 10,
    ("2", "exp"): 120
}

# ================== load response ==================
ResponseMatrix = np.loadtxt(ResponseMatrixFile)

ResponseMatrix = ResponseMatrix / np.max(
    ResponseMatrix,
    axis=1,
    keepdims=True
)

VecOfLambdas = np.linspace(MinLambda, MaxLambda, NumWavelengths) / MaxLambda
VecOfLambdasPlot_nm = VecOfLambdas * MaxLambda

NumRow = ResponseMatrix.shape[0]

ResponseMatrixInterp = np.zeros((NumRow, NumWavelengths))

for r in range(NumRow):
    spline = interpolate.interp1d(
        VecOfLambdas * MaxLambda,
        ResponseMatrix[r, :],
        kind='cubic',
        bounds_error=False,
        fill_value=0.0
    )
    ResponseMatrixInterp[r, :] = spline(VecOfLambdasPlot_nm)

# ================== gaussian ==================
def build_gaussian(FWHM):

    centers = np.linspace(MinLambda, MaxLambda, NumGaussianBasis) / MaxLambda
    sigma = FWHM / (MaxLambda * (2 * np.sqrt(2 * np.log(2))))

    G = np.zeros((NumWavelengths, NumGaussianBasis))

    for j in range(NumGaussianBasis):
        G[:, j] = np.exp(
            -0.5 * ((VecOfLambdas - centers[j]) / sigma) ** 2
        )

    return G

# ================== GCV ==================
def find_gamma(c, A):

    U, s, Vt = svd(A, full_matrices=False)
    m = A.shape[0]
    s = np.maximum(s, 1e-12)

    def gcv(logg):
        g = np.exp(logg)
        fi = s**2 / (s**2 + g**2)
        Utc = U.T @ c

        num = np.sum(((1 - fi) * Utc)**2)
        den = (m - np.sum(fi))**2

        return 1e100 if den < 1e-12 else num / den

    try:
        res = fmin_slsqp(lambda x: gcv(x[0]), x0=[-20], bounds=[(-50, 50)], disp=False)
        return float(np.exp(res[0]))
    except:
        return 1e-6

# ================== reconstruct ==================
def reconstruct(y, FWHM):

    G = build_gaussian(FWHM)
    W = ResponseMatrixInterp @ G
    W = W / np.max(W)

    L = np.eye(NumGaussianBasis)

    if y.ndim == 1:
        y = y[:, np.newaxis]

    N = y.shape[1]
    out = np.zeros((NumWavelengths, N))

    for i in range(N):

        yi = y[:, i]
        if np.max(yi) > 0:
            yi = yi / np.max(yi)

        gamma = find_gamma(yi, W)

        A = np.vstack([W, np.sqrt(gamma) * L])
        b = np.concatenate([yi, np.zeros(L.shape[0])])

        c, _ = nnls(A, b)

        r = G @ c
        if np.max(r) > 0:
            r = r / np.max(r)

        out[:, i] = r

    return out

# ================== PLOT (2 SUBPLOTS) ==================
def plot_two(result, ref, wl, title):

    plt.figure(figsize=(10, 5))

    colors = ['b', 'r']
    labels = ['Ideal', 'Experiment']

    # ===== reconstructed (2 columns) =====
    for i in range(2):

        plt.plot(
            VecOfLambdasPlot_nm,
            result[:, i],
            color=colors[i],
            linewidth=1.8,
            label=f'{labels[i]}'
        )

    # ===== reference (ONLY ONE COLUMN SAFE FIX) =====
    ref_curve = ref[:, 0]   

    if np.max(ref_curve) > 0:
        ref_curve = ref_curve / np.max(ref_curve)

    plt.plot(
        wl,
        ref_curve,
        '--k',
        linewidth=1.2,
        alpha=0.7,
        label='Reference'
    )

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized intensity')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

# ================== RUN ==================
script_folder = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for d in ["1", "2"]:

    Measured = np.loadtxt(MeasuredFiles[d])
    RefData = np.loadtxt(ReferenceFiles[d])

    wl = RefData[:, 0]
    ref = RefData[:, 1:]

    ideal = Measured[:, 0]
    exp = Measured[:, 1]

    recon_ideal = reconstruct(ideal, FWHM_map[(d, "ideal")])
    recon_exp = reconstruct(exp, FWHM_map[(d, "exp")])

    result = np.zeros((NumWavelengths, 2))
    result[:, 0] = recon_ideal[:, 0]
    result[:, 1] = recon_exp[:, 0]

    # plot (ONLY TWO FIGURES TOTAL)
    plot_two(result, ref, wl, f"Reconstruction {d}")

    # save
    out = os.path.join(script_folder, f"recon_{d}_{timestamp}.txt")

    np.savetxt(
        out,
        np.column_stack((VecOfLambdasPlot_nm, result)),
        fmt='%.6f'
    )

    print("Saved:", out)
