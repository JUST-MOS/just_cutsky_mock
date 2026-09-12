"""End-to-end mock generation test/example.

Before running, place the known data files at:
  example/data/uchuu/uchuu_catalog.h5
  py/mockgen/data/nyu_vagc_calibration.npz

Run from the repository root:
  python example/test.py

A successful run writes example/test_mock.h5. That generated mock is the test.
"""

from pathlib import Path
import sys

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from just_cutsky_mock import MockConfig, SkyRegion, generate_mock

UCHUU = ROOT / "example" / "data" / "uchuu" / "uchuu_catalog_mini.h5"
CALIBRATION = ROOT / "py" / "just_cutsky_mock" / "data" / "calibration_table.npz"
OUTPUT = ROOT / "example" / "test_mock.h5"

if not UCHUU.exists():
    raise FileNotFoundError(f"Example host halo catalog is missing:\n  {UCHUU}")
if not CALIBRATION.exists():
    raise FileNotFoundError(f"Precomputed calibration table is missing:\n  {CALIBRATION}")

sky = SkyRegion.centered_area(center_ra=150.0, center_dec=2.0, area_deg2=300.0)

config = MockConfig(
    uchuu_path=UCHUU,
    output_path=OUTPUT,
    sky=sky,
    z_min=0.0,
    z_max=1.0,
    healpix_nside=128,
    random_seed=24,
    n_jobs=-1,
)

result = generate_mock(config)

with h5py.File(result.path, "r") as f:
    assert "galaxies" in f and "halos" in f and "geometry" in f
    assert len(f["galaxies/idx"]) == len(f["halos/idx"])
    assert len(f["galaxies/idx"]) == result.n_galaxies
    assert (f["galaxies/index"][:] == f["halos/index"][:]).all()
    assert len(f["geometry/pixels"]) == result.n_healpix_pixels
    assert f["geometry"].attrs["nside"] == config.healpix_nside
    assert f["geometry"].attrs["ordering"] == "RING"

print(
    f"PASS: generated {result.n_galaxies:,} galaxies in "
    f"{result.n_healpix_pixels:,} HEALPix pixels -> {result.path}"
)
