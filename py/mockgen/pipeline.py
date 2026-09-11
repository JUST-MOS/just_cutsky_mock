from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from .config import BOX_SIZE, H0, MS_THRESHOLD, OM0, MockConfig
from .data import load_nyu_vagc, load_uchuu_catalog
from .hod import populate_hod
from .lightcone import (
    apply_evolution,
    build_corresponding_halo_lightcone,
    build_galaxy_lightcone,
)
from .output import write_mock
from .properties import assign_galaxy_properties


@dataclass(frozen=True)
class MockResult:
    path: Path
    n_galaxies: int
    n_halo_rows: int
    elapsed_seconds: float


def generate_mock(config: MockConfig) -> MockResult:
    """Run the complete fixed-physics cutsky mock pipeline."""
    start = time.time()

    print("[1/6] Loading Uchuu halo catalog...")
    halos = load_uchuu_catalog(config.uchuu_path)
    print(f"      {halos.size:,} halos")

    print("[2/6] Loading NYU-VAGC calibration sample...")
    obs = load_nyu_vagc(config.nyu_vagc_path)
    print(f"      {len(obs.z):,} usable observed galaxies")

    print("[3/6] Populating the periodic box with the fixed HOD model...")
    galaxy_box = populate_hod(
        halos,
        ms_threshold=MS_THRESHOLD,
        box_size=BOX_SIZE,
        random_seed=config.random_seed,
        chunk_size=config.hod_chunk_size,
        n_jobs=config.n_jobs,
    )
    print(f"      {galaxy_box.size:,} HOD galaxies")

    print("[4/6] Assigning halo quenching, g-r color, and Mr...")
    galaxy_box = assign_galaxy_properties(galaxy_box, obs, random_seed=config.random_seed)

    print("[5/6] Building the requested cutsky lightcone and applying evolution...")
    gal_lc, distance_table = build_galaxy_lightcone(
        galaxy_box,
        config.sky,
        h0=H0,
        om0=OM0,
        box_size=BOX_SIZE,
        z_min=config.z_min,
        z_max=config.z_max,
        n_jobs=config.n_jobs,
    )
    gal_lc = apply_evolution(gal_lc, random_seed=config.random_seed)
    print(f"      {len(gal_lc['idx']):,} galaxies after lightcone selection/evolution")

    print("[6/6] Building row-aligned corresponding host halos and writing mock.h5...")
    halo_lc = build_corresponding_halo_lightcone(
        gal_lc,
        halos,
        distance_table,
        box_size=BOX_SIZE,
    )
    write_mock(config.output_path, gal_lc, halo_lc, config)

    elapsed = time.time() - start
    print(f"Done: {config.output_path}")
    print(f"Elapsed: {elapsed:.2f} s")
    return MockResult(
        path=config.output_path,
        n_galaxies=len(gal_lc["idx"]),
        n_halo_rows=len(halo_lc["idx"]),
        elapsed_seconds=elapsed,
    )
