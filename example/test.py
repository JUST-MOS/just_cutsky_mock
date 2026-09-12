"""End-to-end example/test.

Put the lightweight host halo catalog at:
  example/data/uchuu/uchuu_catalog.h5

Then run from the repository root:
  python example/test.py

A successful run writes example/test_mock.h5. That is the test.
"""

from pathlib import Path
import sys

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from mockgen import MockConfig, SkyRegion, generate_mock

UCHUU = ROOT / "example" / "data" / "uchuu" / "uchuu_catalog.h5"
OUTPUT = ROOT / "example" / "test_mock.h5"

if not UCHUU.exists():
    raise FileNotFoundError(f"Example host halo catalog is missing:\n  {UCHUU}")

# Example: a 300 deg^2 spherical cap centered at (RA, Dec) = (150, 2) deg.
sky = SkyRegion.centered_area(center_ra=150.0, center_dec=2.0, area_deg2=300.0)

config = MockConfig(
    uchuu_path=UCHUU,
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
    assert (f["galaxies/index"][:] == f["halos/index"][:]).all()

print(f"PASS: generated {result.n_galaxies:,} galaxies -> {result.path}")
