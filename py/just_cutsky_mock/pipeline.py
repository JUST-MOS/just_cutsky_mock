from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from .calibration import load_calibration
from .config import BOX_SIZE, H0, MS_THRESHOLD, OM0, MockConfig
from .data import load_uchuu_catalog
from .geometry import HealpixGeometry
from .hod import populate_hod
from .lightcone import apply_evolution, build_corresponding_halo_lightcone, build_galaxy_lightcone
from .output import write_mock
from .properties import assign_galaxy_properties


@dataclass(frozen=True)
class MockResult:
    path: Path
    n_galaxies: int
    n_halo_rows: int
    n_healpix_pixels: int
    elapsed_seconds: float


def generate_mock(config: MockConfig) -> MockResult:
    """Run the complete fixed-physics cutsky mock pipeline."""
    start = time.time()

    print("[1/6] Loading host halo catalog...")
    halos = load_uchuu_catalog(config.uchuu_path)
    print(f"      {halos.size:,} halos")

    print("[2/6] Populating the periodic box with the fixed HOD model...")
    galaxy_box = populate_hod(
        halos,
        ms_threshold=MS_THRESHOLD,
        box_size=BOX_SIZE,
        random_seed=config.random_seed,
        chunk_size=config.hod_chunk_size,
        n_jobs=config.n_jobs,
    )
    print(f"      {galaxy_box.size:,} HOD galaxies")

    print("[3/6] Assigning halo quenching, g-r color, and Mr from calibration...")
    calibration = load_calibration()
    galaxy_box = assign_galaxy_properties(galaxy_box, calibration, random_seed=config.random_seed)

    print("[4/6] Converting the requested sky region to HEALPix geometry...")
    geometry = HealpixGeometry.from_sky_region(config.sky, config.healpix_nside)
    print(
        f"      NSIDE={geometry.nside}, RING, {len(geometry.pixels):,} pixels, "
        f"pixelized area={geometry.area_deg2:.3f} deg^2"
    )

    print("[5/6] Building the HEALPix cutsky lightcone and applying evolution...")
    gal_lc, distance_table, lc_diagnostics = build_galaxy_lightcone(
        galaxy_box,
        geometry,
        h0=H0,
        om0=OM0,
        box_size=BOX_SIZE,
        z_min=config.z_min,
        z_max=config.z_max,
        n_jobs=config.n_jobs,
    )
    gal_lc = apply_evolution(gal_lc, random_seed=config.random_seed)
    print(
        f"      {lc_diagnostics['n_candidate_tiles']:.0f} candidate periodic boxes; "
        f"{len(gal_lc['idx']):,} galaxies after selection/evolution"
    )

    print("[6/6] Building row-aligned host halos and writing mock.h5...")
    halo_lc = build_corresponding_halo_lightcone(gal_lc, halos, distance_table, box_size=BOX_SIZE)
    write_mock(config.output_path, gal_lc, halo_lc, geometry, config, lc_diagnostics)

    elapsed = time.time() - start
    print(f"Done: {config.output_path}")
    print(f"Elapsed: {elapsed:.2f} s")
    return MockResult(
        path=config.output_path,
        n_galaxies=len(gal_lc["idx"]),
        n_halo_rows=len(halo_lc["idx"]),
        n_healpix_pixels=len(geometry.pixels),
        elapsed_seconds=elapsed,
    )
