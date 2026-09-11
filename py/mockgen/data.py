from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class HaloBox:
    mass: np.ndarray
    conc: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    vrms: np.ndarray
    index: np.ndarray

    @property
    def size(self) -> int:
        return len(self.mass)


@dataclass
class ObservationalSample:
    z: np.ndarray
    Mr: np.ndarray
    gr: np.ndarray
    Ms: np.ndarray
    z_mag: np.ndarray
    Mr_mag: np.ndarray
    gr_mag: np.ndarray
    Ms_mag: np.ndarray


@dataclass
class GalaxyBox:
    Ms: np.ndarray              # log10 stellar mass
    Mh: np.ndarray              # host halo mass, linear
    pos: np.ndarray
    vel: np.ndarray
    type: np.ndarray            # 1 central, 0 satellite
    index: np.ndarray           # source halo row index
    gr: np.ndarray | None = None
    Mr: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self.Ms)


UCHUU_REQUIRED_FIELDS = (
    "halo_mass",
    "halo_conc",
    "halo_x",
    "halo_y",
    "halo_z",
    "halo_vx",
    "halo_vy",
    "halo_vz",
    "halo_vrms",
)


def load_uchuu_catalog(path: str | Path) -> HaloBox:
    """Load the fixed Uchuu schema used by this project.

    The raw HDF5 dataset names are intentionally isolated in this function so
    the physical pipeline never depends directly on them.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Uchuu catalog not found: {path}")

    with h5py.File(path, "r") as f:
        missing = [name for name in UCHUU_REQUIRED_FIELDS if name not in f]
        if missing:
            raise KeyError(f"Missing required Uchuu datasets: {missing}")

        mass = f["halo_mass"][:]
        conc = f["halo_conc"][:]
        pos = np.column_stack((f["halo_x"][:], f["halo_y"][:], f["halo_z"][:]))
        vel = np.column_stack((f["halo_vx"][:], f["halo_vy"][:], f["halo_vz"][:]))
        vrms = f["halo_vrms"][:]

    n = len(mass)
    if not (len(conc) == len(pos) == len(vel) == len(vrms) == n):
        raise ValueError("Inconsistent array lengths in Uchuu catalog.")

    return HaloBox(
        mass=np.asarray(mass),
        conc=np.asarray(conc),
        pos=np.asarray(pos),
        vel=np.asarray(vel),
        vrms=np.asarray(vrms),
        index=np.arange(n, dtype=np.int64),
    )


def stellar_mass_limit(z):
    return 5.4 * np.maximum(np.asarray(z) - 0.025, 0.0) ** 0.33 + 8.0


def vlim(lgMs):
    lgMs = np.asarray(lgMs)
    return np.where(lgMs > 8.0, ((lgMs - 8.0) / 5.4) ** (1.0 / 0.33) + 0.025, 0.025)


def load_nyu_vagc(path: str | Path) -> ObservationalSample:
    """Load the fixed NYU-VAGC ``all0.dat`` column convention.

    Columns used by the original implementation:
      3 -> redshift
      4 -> Mr
      7 -> g-r
      8 -> log10 stellar mass
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NYU-VAGC file not found: {path}")

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

    zlo, zhi = 0.08, 0.12
    mask_mag = (
        np.isfinite(Mr)
        & np.isfinite(gr)
        & np.isfinite(Ms)
        & (z > zlo)
        & (z < zhi)
        & (Ms > stellar_mass_limit(zhi))
        & (Ms < 12.0)
    )

    return ObservationalSample(
        z=z,
        Mr=Mr,
        gr=gr,
        Ms=Ms,
        z_mag=z[mask_mag],
        Mr_mag=Mr[mask_mag],
        gr_mag=gr[mask_mag],
        Ms_mag=Ms[mask_mag],
    )
