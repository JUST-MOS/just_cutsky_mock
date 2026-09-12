"""Fit the NYU-VAGC calibration table used by the runtime pipeline.

This script is intentionally separate from mock generation. Run it once on the
full all0.dat, inspect the fitted values, and commit the resulting NPZ file.
Runtime mock generation then needs only uchuu_catalog.h5.

Usage
-----
python py/fit_calibration.py /path/to/all0.dat

Optional output path:
python py/fit_calibration.py /path/to/all0.dat --output /path/to/table.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mockgen.calibration import COLOR_MASS_EDGES, CALIBRATION_FILENAME


def stellar_mass_limit(z):
    z = np.asarray(z)
    return 5.4 * np.maximum(z - 0.025, 0.0) ** 0.33 + 8.0


def vlim(lg_ms):
    lg_ms = np.asarray(lg_ms)
    return np.where(
        lg_ms > 8.0,
        ((lg_ms - 8.0) / 5.4) ** (1.0 / 0.33) + 0.025,
        0.025,
    )


def double_gaussian(x, a1, mu1, sigma1, mu2, sigma2):
    return (
        a1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) / (sigma1 * np.sqrt(2.0 * np.pi))
        + (1.0 - a1)
        * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
        / (sigma2 * np.sqrt(2.0 * np.pi))
    )


def fit_step(data, bins=25):
    data = np.asarray(data)
    data = data[(data > 0.2) & (data < 1.2)]
    if data.size < 8:
        raise ValueError("Not enough g-r values for the double-Gaussian fit.")
    counts, edges = np.histogram(data, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    popt, _ = curve_fit(
        double_gaussian,
        centers,
        counts,
        p0=[0.5, 0.6, 0.1, 0.9, 0.1],
        bounds=([0.0, 0.5, 0.1, 0.8, 0.05], [1.0, 0.9, 0.3, 1.2, 0.2]),
        maxfev=5000,
    )
    return popt


def load_all0(path: Path):
    all0 = np.loadtxt(path)
    if all0.ndim != 2 or all0.shape[1] <= 8:
        raise ValueError("all0.dat must contain at least 9 columns.")

    z = all0[:, 3]
    Mr = all0[:, 4]
    gr = all0[:, 7]
    Ms = all0[:, 8]

    mask = (
        np.isfinite(Mr)
        & np.isfinite(gr)
        & np.isfinite(Ms)
        & (z > 0.01)
        & ((Ms <= 8.0) | (Ms > stellar_mass_limit(z)))
    )
    z, Mr, gr, Ms = z[mask], Mr[mask], gr[mask], Ms[mask]

    mag_mask = (
        np.isfinite(Mr)
        & np.isfinite(gr)
        & np.isfinite(Ms)
        & (z > 0.08)
        & (z < 0.12)
        & (Ms > stellar_mass_limit(0.12))
        & (Ms < 12.0)
    )

    return z, Mr, gr, Ms, z[mag_mask], Mr[mag_mask], gr[mag_mask], Ms[mag_mask]


def fit_calibration(all0_path: Path):
    z, Mr, gr, Ms, z_mag, Mr_mag, gr_mag, Ms_mag = load_all0(all0_path)
    del z_mag

    global_sample = gr[(gr > 0.2) & (gr < 1.2)]
    if global_sample.size < 8:
        raise ValueError("Not enough valid global g-r values.")

    params = np.empty((len(COLOR_MASS_EDGES) - 1, 5), dtype=float)
    for i, (low, high) in enumerate(zip(COLOR_MASS_EDGES[:-1], COLOR_MASS_EDGES[1:])):
        obs_mask = (Ms > low) & (Ms < high) & (z < vlim(low))
        sample = gr[obs_mask]
        try:
            params[i] = fit_step(sample)
        except (ValueError, RuntimeError):
            # Keep the runtime table complete for sparse edge bins, matching the
            # previous runtime fallback behavior.
            params[i] = fit_step(global_sample)
        print(
            f"bin {i:02d}: {low:.4f} < logM* < {high:.4f}, "
            f"N={sample.size:6d}, params={params[i]}"
        )

    if Ms_mag.size < 3:
        raise ValueError("Not enough objects in the Mr calibration sample.")
    A = np.column_stack((np.ones_like(Ms_mag), Ms_mag, gr_mag))
    coeffs, _, _, _ = np.linalg.lstsq(A, Mr_mag, rcond=None)
    sigma = float(np.std(Mr_mag - A @ coeffs))

    return params, coeffs, sigma


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("all0", type=Path, help="Path to the full NYU-VAGC all0.dat")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "mockgen" / "data" / CALIBRATION_FILENAME,
        help="Output NPZ path",
    )
    args = parser.parse_args(argv)

    params, coeffs, sigma = fit_calibration(args.all0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        mass_edges=COLOR_MASS_EDGES,
        color_params=params,
        mr_coeffs=coeffs,
        mr_sigma=np.array(sigma),
    )

    print("\nMr relation:")
    print(f"  Mr = {coeffs[0]:.10g} + {coeffs[1]:.10g} logM* + {coeffs[2]:.10g} (g-r)")
    print(f"  sigma_Mr = {sigma:.10g}")
    print(f"\nSaved calibration table: {args.output}")


if __name__ == "__main__":
    main()
