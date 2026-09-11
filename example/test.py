"""End-to-end example/test.

Put the lightweight files at:
  examples/data/uchuu/uchuu_catalog.h5
  examples/data/nyu-vagc/all0.dat

Then run:
  python test.py

A successful run writes test_mock.h5. That is the test.
"""

from pathlib import Path
import sys

import h5py

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mockgen import MockConfig, SkyRegion, generate_mock

UCHUU = ROOT / "example" / "data" / "uchuu" / "uchuu_catalog.h5"
NYU = ROOT / "example" / "data" / "nyu-vagc" / "all0.dat"
OUTPUT = ROOT / "test_mock.h5"

if not UCHUU.exists() or not NYU.exists():
    raise FileNotFoundError(
        "Example data are missing. Expected:\n"
        f"  {UCHUU}\n"
        f"  {NYU}\n"
    )

# Example: a 300 deg^2 spherical cap centered at (RA, Dec) = (150, 2) deg.
sky = SkyRegion.centered_area(center_ra=150.0, center_dec=2.0, area_deg2=300.0)

config = MockConfig(
    uchuu_path=UCHUU,
    nyu_vagc_path=NYU,
    output_path=OUTPUT,
    sky=sky,
    z_min=0.0,
    z_max=1.0,
    random_seed=24,
    n_jobs=-1,
)

result = generate_mock(config)

with h5py.File(result.path, "r") as f:
    assert "galaxies" in f and "halos" in f
    assert len(f["galaxies/idx"]) == len(f["halos/idx"])
    assert len(f["galaxies/idx"]) == result.n_galaxies

print(f"PASS: generated {result.n_galaxies:,} galaxies -> {result.path}")
