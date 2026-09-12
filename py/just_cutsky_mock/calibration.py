from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np


CALIBRATION_FILENAME = "nyu_vagc_calibration.npz"


@dataclass(frozen=True)
class GalaxyCalibration:
    mass_edges: np.ndarray
    color_params: np.ndarray
    mr_coeffs: np.ndarray
    mr_sigma: float

    @property
    def n_bins(self) -> int:
        return len(self.mass_edges) - 1


def default_calibration_path() -> Path:
    return Path(files("just_cutsky_mock").joinpath("data", CALIBRATION_FILENAME))


def load_calibration(path: str | Path | None = None) -> GalaxyCalibration:
    path = default_calibration_path() if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "Precomputed galaxy calibration table is missing. Expected:\n"
            f"  {path}\n"
            "Place the known nyu_vagc_calibration.npz at that package-data path."
        )

    with np.load(path) as table:
        required = {"mass_edges", "color_params", "mr_coeffs", "mr_sigma"}
        missing = required.difference(table.files)
        if missing:
            raise KeyError(f"Calibration table is missing arrays: {sorted(missing)}")
        mass_edges = np.asarray(table["mass_edges"], dtype=float)
        color_params = np.asarray(table["color_params"], dtype=float)
        mr_coeffs = np.asarray(table["mr_coeffs"], dtype=float)
        mr_sigma = float(np.asarray(table["mr_sigma"]).reshape(()))

    if color_params.shape != (len(mass_edges) - 1, 5):
        raise ValueError("color_params must have shape (len(mass_edges)-1, 5).")
    if mr_coeffs.shape != (3,):
        raise ValueError("mr_coeffs must have shape (3,).")
    if not np.all(np.isfinite(color_params)):
        raise ValueError("Calibration color_params contains non-finite values.")
    if not np.all(np.isfinite(mr_coeffs)) or not np.isfinite(mr_sigma):
        raise ValueError("Calibration Mr relation contains non-finite values.")

    return GalaxyCalibration(
        mass_edges=mass_edges,
        color_params=color_params,
        mr_coeffs=mr_coeffs,
        mr_sigma=mr_sigma,
    )
