from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .config import BOX_SIZE, H0, HDF5_COMPRESSION, OM0, MockConfig


GALAXY_FIELDS = (
    "idx",
    "index",
    "type",
    "ra",
    "dec",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "z_obs",
    "z_com",
    "Mh",
    "Ms",
    "Mr",
    "mr",
    "gr",
)

HALO_FIELDS = (
    "idx",
    "index",
    "ra",
    "dec",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "z_obs",
    "z_com",
    "Mh",
    "conc",
    "vrms",
)

FIELD_ATTRS = {
    "idx": {"description": "Row ID within this output catalog."},
    "index": {"description": "Row index of the source halo in uchuu_catalog.h5."},
    "type": {"description": "Galaxy type: 1 central, 0 satellite."},
    "ra": {"unit": "deg"},
    "dec": {"unit": "deg"},
    "x": {"unit": "Mpc/h"},
    "y": {"unit": "Mpc/h"},
    "z": {"unit": "Mpc/h"},
    "vx": {"unit": "km/s"},
    "vy": {"unit": "km/s"},
    "vz": {"unit": "km/s"},
    "z_obs": {"description": "Observed redshift including peculiar velocity."},
    "z_com": {"description": "Cosmological redshift from comoving distance."},
    "Mh": {"unit": "Msun/h", "description": "Linear host halo M200m."},
    "Ms": {"description": "log10 stellar mass, following the reference implementation."},
    "Mr": {"unit": "mag", "description": "Rest-frame r-band absolute magnitude after evolution."},
    "mr": {"unit": "mag", "description": "Apparent r-band magnitude after evolution."},
    "gr": {"unit": "mag", "description": "g-r color."},
    "conc": {"description": "Host halo C200m concentration from uchuu_catalog.h5."},
    "vrms": {"unit": "km/s", "description": "Host halo velocity RMS from uchuu_catalog.h5."},
}


def _write_group(group: h5py.Group, data: dict[str, np.ndarray], fields, compression):
    for name in fields:
        if name not in data:
            raise KeyError(f"Output field {name!r} is missing.")
        dset = group.create_dataset(name, data=data[name], compression=compression)
        for key, value in FIELD_ATTRS.get(name, {}).items():
            dset.attrs[key] = value


def write_mock(
    path: str | Path,
    galaxies: dict[str, np.ndarray],
    halos: dict[str, np.ndarray],
    config: MockConfig,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    # The internal tile field is deliberately not part of the public schema.
    public_galaxies = {k: galaxies[k] for k in GALAXY_FIELDS}

    with h5py.File(path, "w") as f:
        f.attrs["format"] = "galaxy + row-aligned corresponding host halo lightcone"
        f.attrs["random_seed"] = config.random_seed
        f.attrs["H0"] = H0
        f.attrs["Om0"] = OM0
        f.attrs["h"] = H0 / 100.0
        f.attrs["box_size_Mpc_h"] = BOX_SIZE
        f.attrs["z_min"] = config.z_min
        f.attrs["z_max"] = config.z_max
        f.attrs["sky_region"] = config.sky.describe()
        f.attrs["halo_alignment"] = "/halos[i] is the host halo of /galaxies[i]"

        ggal = f.create_group("galaxies")
        ghalo = f.create_group("halos")
        ggal.attrs["n_objects"] = len(public_galaxies["idx"])
        ghalo.attrs["n_objects"] = len(halos["idx"])
        ghalo.attrs["row_aligned_with"] = "/galaxies"

        _write_group(ggal, public_galaxies, GALAXY_FIELDS, HDF5_COMPRESSION)
        _write_group(ghalo, halos, HALO_FIELDS, HDF5_COMPRESSION)
