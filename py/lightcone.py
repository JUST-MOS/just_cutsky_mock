from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.integrate import cumulative_trapezoid
from joblib import Parallel, delayed
from tqdm import tqdm

from .config import SkyRegion
from .data import GalaxyBox, HaloBox
from .models import delta_mag_function, weight_function


C_KM_S = 299792.458


@dataclass
class DistanceTable:
    h: float
    z_samples: np.ndarray
    dcom_h: np.ndarray
    dlum_mpc: np.ndarray

    @classmethod
    def build(cls, h0: float, om0: float, z_max: float):
        # Flat-LambdaCDM distance table, numerically equivalent to the original
        # FlatLambdaCDM(H0=67.74, Om0=0.3089) use for this pipeline.
        table_zmax = max(5.0, z_max + 0.5)
        z_samples = np.arange(0.0, table_zmax + 0.001, 0.001)
        h = h0 / 100.0
        ez = np.sqrt(om0 * (1.0 + z_samples) ** 3 + (1.0 - om0))
        dcom_mpc = (C_KM_S / h0) * cumulative_trapezoid(1.0 / ez, z_samples, initial=0.0)
        dcom_h = dcom_mpc * h
        dlum_mpc = (1.0 + z_samples) * dcom_mpc
        return cls(h=h, z_samples=z_samples, dcom_h=dcom_h, dlum_mpc=dlum_mpc)

    def z_from_dcom_h(self, distance):
        return np.interp(distance, self.dcom_h, self.z_samples)

    def dcom_h_from_z(self, z):
        return np.interp(z, self.z_samples, self.dcom_h)

    def dlum_from_z(self, z):
        return np.interp(z, self.z_samples, self.dlum_mpc)


def _unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _angular_sep_vectors_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _aabb_radial_bounds(tile_xyz: tuple[int, int, int], box_size: float) -> tuple[float, float, np.ndarray, float]:
    lo = np.asarray(tile_xyz, dtype=float) * box_size
    hi = lo + box_size

    # Minimum distance from origin to an axis-aligned box.
    closest = np.where(0.0 < lo, lo, np.where(0.0 > hi, hi, 0.0))
    dmin = float(np.linalg.norm(closest))

    # Maximum distance occurs at a corner with maximum absolute coordinate on each axis.
    far = np.maximum(np.abs(lo), np.abs(hi))
    dmax = float(np.linalg.norm(far))

    center = 0.5 * (lo + hi)
    radius = np.sqrt(3.0) * box_size / 2.0
    return dmin, dmax, center, radius


def candidate_tiles(
    sky: SkyRegion,
    *,
    box_size: float,
    dmin: float,
    dmax: float,
) -> list[tuple[int, int, int]]:
    """Find periodic replicas that can intersect the requested cutsky volume."""
    n = int(np.ceil(dmax / box_size)) + 1
    cap_ra, cap_dec, cap_radius_deg = sky.bounding_cap()
    cap_vec = _unit_vector(cap_ra, cap_dec)

    selected = []
    for tile in product(range(-n, n + 1), repeat=3):
        box_dmin, box_dmax, center, sphere_radius = _aabb_radial_bounds(tile, box_size)
        if box_dmax < dmin or box_dmin > dmax:
            continue

        center_dist = float(np.linalg.norm(center))
        if center_dist <= sphere_radius or cap_radius_deg >= 179.999:
            selected.append(tile)
            continue

        box_ang_radius = float(np.degrees(np.arcsin(np.clip(sphere_radius / center_dist, 0.0, 1.0))))
        center_vec = center / center_dist
        sep = _angular_sep_vectors_deg(center_vec, cap_vec)
        if sep <= cap_radius_deg + box_ang_radius:
            selected.append(tile)

    return selected


def _empty_galaxy_lightcone() -> dict[str, np.ndarray]:
    scalar = ("index", "type", "ra", "dec", "x", "y", "z", "vx", "vy", "vz", "z_obs", "z_com", "Mh", "Ms", "Mr", "mr", "gr")
    out = {k: np.empty(0) for k in scalar}
    out["index"] = np.empty(0, dtype=np.int64)
    out["type"] = np.empty(0, dtype=np.int8)
    out["tile"] = np.empty((0, 3), dtype=np.int32)
    return out


def _process_tile(
    tile: tuple[int, int, int],
    galaxies: GalaxyBox,
    sky: SkyRegion,
    distances: DistanceTable,
    box_size: float,
    radial_dmin: float,
    radial_dmax: float,
    z_min: float,
    z_max: float,
) -> dict[str, np.ndarray]:
    offset = np.asarray(tile, dtype=float) * box_size
    pos = galaxies.pos + offset
    r = np.linalg.norm(pos, axis=1)
    mask = (r > 0.0) & (r >= radial_dmin) & (r <= radial_dmax)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return _empty_galaxy_lightcone()

    pos = pos[idx]
    r = r[idx]
    ra = np.mod(np.degrees(np.arctan2(pos[:, 1], pos[:, 0])), 360.0)
    dec = np.degrees(np.arcsin(np.clip(pos[:, 2] / r, -1.0, 1.0)))
    angular = sky.contains(ra, dec)
    idx2 = np.flatnonzero(angular)
    if idx2.size == 0:
        return _empty_galaxy_lightcone()

    source = idx[idx2]
    pos = pos[idx2]
    r = r[idx2]
    ra = ra[idx2]
    dec = dec[idx2]
    vel = galaxies.vel[source]

    v_radial = np.sum(pos * vel, axis=1) / r
    beta = np.clip(v_radial / C_KM_S, -0.999999, 0.999999)
    z_com = distances.z_from_dcom_h(r)
    z_obs = (1.0 + z_com) / (1.0 - beta) - 1.0

    zmask = (z_obs > z_min) & (z_obs < z_max)
    source = source[zmask]
    pos = pos[zmask]
    vel = vel[zmask]
    ra = ra[zmask]
    dec = dec[zmask]
    z_com = z_com[zmask]
    z_obs = z_obs[zmask]
    if source.size == 0:
        return _empty_galaxy_lightcone()

    Mr = galaxies.Mr[source]
    dlum_mpc = distances.dlum_from_z(z_com)
    dm = 5.0 * np.log10(dlum_mpc) + 25.0
    mr = (
        Mr
        + 5.0 * np.log10(distances.h)
        + 2.5 * np.log10((z_obs + 0.9) / 1.1)
        + dm
        - 1.62 * (z_com - 0.1)
    )

    return {
        "index": galaxies.index[source].astype(np.int64, copy=False),
        "type": galaxies.type[source].astype(np.int8, copy=False),
        "ra": ra,
        "dec": dec,
        "x": pos[:, 0],
        "y": pos[:, 1],
        "z": pos[:, 2],
        "vx": vel[:, 0],
        "vy": vel[:, 1],
        "vz": vel[:, 2],
        "z_obs": z_obs,
        "z_com": z_com,
        "Mh": galaxies.Mh[source],
        "Ms": galaxies.Ms[source],
        "Mr": Mr,
        "mr": mr,
        "gr": galaxies.gr[source],
        "tile": np.repeat(np.asarray(tile, dtype=np.int32)[None, :], source.size, axis=0),
    }


def build_galaxy_lightcone(
    galaxies: GalaxyBox,
    sky: SkyRegion,
    *,
    h0: float,
    om0: float,
    box_size: float,
    z_min: float,
    z_max: float,
    n_jobs: int,
) -> tuple[dict[str, np.ndarray], DistanceTable]:
    distances = DistanceTable.build(h0, om0, z_max)

    # Buffer is only used for geometric preselection; the final cut is on z_obs.
    z_buffer = 0.05
    radial_dmin = distances.dcom_h_from_z(max(0.0, z_min - z_buffer))
    radial_dmax = distances.dcom_h_from_z(z_max + z_buffer)
    tiles = candidate_tiles(sky, box_size=box_size, dmin=radial_dmin, dmax=radial_dmax)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_tile)(
            tile,
            galaxies,
            sky,
            distances,
            box_size,
            radial_dmin,
            radial_dmax,
            z_min,
            z_max,
        )
        for tile in tqdm(tiles, desc="Cutsky lightcone")
    )

    results = [r for r in results if len(r["z_obs"]) > 0]
    if not results:
        return _empty_galaxy_lightcone(), distances

    keys = ["index", "type", "ra", "dec", "x", "y", "z", "vx", "vy", "vz", "z_obs", "z_com", "Mh", "Ms", "Mr", "mr", "gr", "tile"]
    return {k: np.concatenate([r[k] for r in results], axis=0) for k in keys}, distances


def apply_evolution(lightcone: dict[str, np.ndarray], *, random_seed: int) -> dict[str, np.ndarray]:
    if len(lightcone["z_com"]) == 0:
        lightcone["idx"] = np.empty(0, dtype=np.int64)
        return lightcone

    rng = np.random.default_rng(np.random.SeedSequence([random_seed, 4001]))
    z = lightcone["z_com"]
    dmag = delta_mag_function(z)
    keep = rng.random(len(z), dtype=np.float32) < weight_function(z)

    out = {}
    for key, value in lightcone.items():
        out[key] = value[keep]
    out["Mr"] = out["Mr"] + dmag[keep]
    out["mr"] = out["mr"] + dmag[keep]
    out["idx"] = np.arange(len(out["z_com"]), dtype=np.int64)
    return out


def build_corresponding_halo_lightcone(
    galaxy_lightcone: dict[str, np.ndarray],
    halos: HaloBox,
    distances: DistanceTable,
    *,
    box_size: float,
) -> dict[str, np.ndarray]:
    """Build the row-aligned host-halo companion catalog.

    Row i in this catalog is the host halo of row i in the galaxy catalog.
    Host halos therefore intentionally repeat when several galaxies share one halo.
    """
    n = len(galaxy_lightcone["index"])
    if n == 0:
        return {
            "idx": np.empty(0, dtype=np.int64),
            "index": np.empty(0, dtype=np.int64),
            "ra": np.empty(0),
            "dec": np.empty(0),
            "x": np.empty(0),
            "y": np.empty(0),
            "z": np.empty(0),
            "vx": np.empty(0),
            "vy": np.empty(0),
            "vz": np.empty(0),
            "z_obs": np.empty(0),
            "z_com": np.empty(0),
            "Mh": np.empty(0),
            "conc": np.empty(0),
            "vrms": np.empty(0),
        }

    source = galaxy_lightcone["index"].astype(np.int64, copy=False)
    tile = galaxy_lightcone["tile"]
    pos = halos.pos[source] + tile.astype(float) * box_size
    vel = halos.vel[source]
    r = np.linalg.norm(pos, axis=1)
    safe_r = np.where(r == 0.0, 1.0e-10, r)

    ra = np.mod(np.degrees(np.arctan2(pos[:, 1], pos[:, 0])), 360.0)
    dec = np.degrees(np.arcsin(np.clip(pos[:, 2] / safe_r, -1.0, 1.0)))
    v_radial = np.sum(pos * vel, axis=1) / safe_r
    beta = np.clip(v_radial / C_KM_S, -0.999999, 0.999999)
    z_com = distances.z_from_dcom_h(r)
    z_obs = (1.0 + z_com) / (1.0 - beta) - 1.0

    return {
        "idx": galaxy_lightcone["idx"].astype(np.int64, copy=False),
        "index": source,
        "ra": ra,
        "dec": dec,
        "x": pos[:, 0],
        "y": pos[:, 1],
        "z": pos[:, 2],
        "vx": vel[:, 0],
        "vy": vel[:, 1],
        "vz": vel[:, 2],
        "z_obs": z_obs,
        "z_com": z_com,
        "Mh": halos.mass[source],
        "conc": halos.conc[source],
        "vrms": halos.vrms[source],
    }
