# The aim of this script is to generate a mini galaxy mock called "uchuu_mini_mock.h5" in the "example" directory
# This script will use the "example/simulation/uchuu_mini.h5" host halo catalog to generate the mock
# Mention: "example/simulation/uchuu_mini.h5" is not the complete Uchuu host halo catalog, 
# due to the large size of Uchuu, this file only contains a small subset of the complete catalog, the file size is about 15MB

from pathlib import Path
import sys
import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from just_cutsky_mock import MockConfig, SkyRegion, generate_mock

UCHUU = ROOT / "example" / "simulaion" / "uchuu_mini.h5"
CALIBRATION = ROOT / "py" / "just_cutsky_mock" / "data" / "calibration_table.npz"
OUTPUT = ROOT / "example" / "uchuu_mini_mock.h5"

if not UCHUU.exists():
    raise FileNotFoundError(f"Example host halo catalog is missing:\n  {UCHUU}")
if not CALIBRATION.exists():
    raise FileNotFoundError(f"Precomputed calibration table is missing:\n  {CALIBRATION}")

# define the target sky region, the following code will generate galaxy mock in a circle region with area=100Deg^2 centered at (ra, dec) = (45, 45)
sky = SkyRegion.centered_area(center_ra=45.0, center_dec=45.0, area_deg2=100.0)
# if you want to generate a mock in a geometrically limited area (e.g., ra in [0, 90] and dec in [0, 90]), define "sky" through:
# sky = SkyRegion.rectangle(ra_min=0.0, ra_max=90.0, dec_min=0.0, dec_max=90.0)

config = MockConfig(
    halo_path=UCHUU,
    output_path=OUTPUT,
    sky=sky,
    z_min=0.0,
    z_max=0.5, # generate a galaxy mock with redshift range from 0 to 0.5
    healpix_nside=128,
    random_seed=24,
    n_jobs=-1)

result = generate_mock(config)
print(f"PASS: generated {result.n_galaxies:,} galaxies in {result.n_healpix_pixels:,} HEALPix pixels -> {result.path}")
