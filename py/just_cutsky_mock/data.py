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
    """Load the fixed host-halo input schema used by the pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Host halo catalog not found: {path}")

    with h5py.File(path, "r") as f:
        missing = [name for name in UCHUU_REQUIRED_FIELDS if name not in f]
        if missing:
            raise KeyError(f"Missing required host-halo datasets: {missing}")

        mass = f["halo_mass"][:]
        conc = f["halo_conc"][:]
        pos = np.column_stack((f["halo_x"][:], f["halo_y"][:], f["halo_z"][:]))
        vel = np.column_stack((f["halo_vx"][:], f["halo_vy"][:], f["halo_vz"][:]))
        vrms = f["halo_vrms"][:]

    n = len(mass)
    if not (len(conc) == len(pos) == len(vel) == len(vrms) == n):
        raise ValueError("Inconsistent array lengths in host halo catalog.")

    return HaloBox(
        mass=np.asarray(mass),
        conc=np.asarray(conc),
        pos=np.asarray(pos),
        vel=np.asarray(vel),
        vrms=np.asarray(vrms),
        index=np.arange(n, dtype=np.int64),
    )
