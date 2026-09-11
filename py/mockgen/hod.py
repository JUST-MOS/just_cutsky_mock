from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .data import GalaxyBox, HaloBox
from .models import FixedHODModel


def _chunk_seed(master_seed: int, chunk_id: int) -> int:
    return int(np.random.SeedSequence([master_seed, 1001, chunk_id]).generate_state(1, dtype=np.uint64)[0])


def _process_chunk(
    halos: HaloBox,
    model: FixedHODModel,
    start: int,
    end: int,
    box_size: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    mass = halos.mass[start:end]
    conc = halos.conc[start:end]
    pos = halos.pos[start:end]
    vel = halos.vel[start:end]
    vrms = halos.vrms[start:end]
    source_index = halos.index[start:end]

    # Centrals: same SHMR/scatter prescription as the reference implementation.
    mu_star = np.log10(model.mean_stellar_mass(mass))
    sig_star = model.shmr_scatter(mass)
    cen_logms_all = rng.normal(mu_star, sig_star)
    has_central = cen_logms_all >= np.log10(model.ms_threshold)
    cen_idx = np.flatnonzero(has_central)

    if len(cen_idx):
        cen_dv = rng.normal(0.0, 0.2, size=(len(cen_idx), 3)) * vrms[cen_idx, None] / np.sqrt(3.0)
        cen = {
            "Ms": cen_logms_all[cen_idx],
            "Mh": mass[cen_idx],
            "pos": pos[cen_idx],
            "vel": vel[cen_idx] + cen_dv,
            "type": np.ones(len(cen_idx), dtype=np.int8),
            "index": source_index[cen_idx].astype(np.int64, copy=False),
        }
    else:
        cen = {
            "Ms": np.empty(0),
            "Mh": np.empty(0),
            "pos": np.empty((0, 3)),
            "vel": np.empty((0, 3)),
            "type": np.empty(0, dtype=np.int8),
            "index": np.empty(0, dtype=np.int64),
        }

    # Satellites.
    lambda_sat = model.expected_num_sat(mass)
    nsat = rng.poisson(lambda_sat)
    parent_idx = np.flatnonzero(nsat > 0)
    counts = nsat[parent_idx]
    total_sat = int(np.sum(counts))

    if total_sat:
        parent_mh = np.repeat(mass[parent_idx], counts)
        parent_conc = np.repeat(conc[parent_idx], counts)
        parent_pos = np.repeat(pos[parent_idx], counts, axis=0)
        parent_vel = np.repeat(vel[parent_idx], counts, axis=0)
        parent_vrms = np.repeat(vrms[parent_idx], counts)
        parent_index = np.repeat(source_index[parent_idx], counts)

        sat_mass = model.sample_satellite_masses(parent_mh, rng.random(total_sat))
        sat_pos = model.sample_nfw_positions(
            parent_mh,
            parent_conc,
            parent_pos,
            rng.random(total_sat),
            rng.random(total_sat),
            rng.random(total_sat),
            box_size,
        )
        sat_dv = rng.normal(0.0, 1.0, size=(total_sat, 3)) * parent_vrms[:, None] / np.sqrt(3.0)
        sat = {
            "Ms": np.log10(sat_mass),
            "Mh": parent_mh,
            "pos": sat_pos,
            "vel": parent_vel + sat_dv,
            "type": np.zeros(total_sat, dtype=np.int8),
            "index": parent_index.astype(np.int64, copy=False),
        }
    else:
        sat = {
            "Ms": np.empty(0),
            "Mh": np.empty(0),
            "pos": np.empty((0, 3)),
            "vel": np.empty((0, 3)),
            "type": np.empty(0, dtype=np.int8),
            "index": np.empty(0, dtype=np.int64),
        }

    return {
        key: np.concatenate((cen[key], sat[key]), axis=0)
        for key in ("Ms", "Mh", "pos", "vel", "type", "index")
    }


def populate_hod(
    halos: HaloBox,
    *,
    ms_threshold: float,
    box_size: float,
    random_seed: int,
    chunk_size: int,
    n_jobs: int,
) -> GalaxyBox:
    """Populate the periodic halo box with centrals and satellites."""
    model = FixedHODModel(ms_threshold=ms_threshold)
    n_chunks = (halos.size + chunk_size - 1) // chunk_size

    jobs = []
    for chunk_id in range(n_chunks):
        start = chunk_id * chunk_size
        end = min((chunk_id + 1) * chunk_size, halos.size)
        jobs.append((chunk_id, start, end))

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_chunk)(
            halos,
            model,
            start,
            end,
            box_size,
            _chunk_seed(random_seed, chunk_id),
        )
        for chunk_id, start, end in tqdm(jobs, desc="HOD population")
    )

    if not results:
        return GalaxyBox(
            Ms=np.empty(0),
            Mh=np.empty(0),
            pos=np.empty((0, 3)),
            vel=np.empty((0, 3)),
            type=np.empty(0, dtype=np.int8),
            index=np.empty(0, dtype=np.int64),
        )

    merged = {
        key: np.concatenate([r[key] for r in results], axis=0)
        for key in ("Ms", "Mh", "pos", "vel", "type", "index")
    }
    return GalaxyBox(**merged)
