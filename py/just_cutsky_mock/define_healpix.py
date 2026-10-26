import h5py
import numpy as np


def read_healpix(path):
    with h5py.File(path, "r") as f:
        pixels = f["geometry/pixels"][:]
        nside = f["geometry"].attrs["nside"]
        ordering = f["geometry"].attrs["ordering"]
    return pixels, nside, ordering


def make_healpix_mask(pixels, nside):
    mask = np.zeros(12 * nside**2, dtype=bool)
    mask[pixels] = True
    return mask
