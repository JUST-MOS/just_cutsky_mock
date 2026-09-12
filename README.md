# cutsky_mock

This repository generates one product only: a galaxy lightcone in a requested sky region together with the corresponding host-halo lightcone.

The runtime scientific input is a single host-halo catalog:

```text
uchuu_catalog.h5
```

The NYU-VAGC sample is **not** a runtime input. Its color and luminosity calibration is fitted once offline and stored in:

```text
py/mockgen/data/nyu_vagc_calibration.npz
```

The final output is a single HDF5 file:

```text
mock.h5
├── /galaxies
└── /halos
```

Row `i` in `/halos` is the host halo of row `i` in `/galaxies`. A host halo is intentionally repeated when several output galaxies share it.

## Repository layout

```text
README.md
pyproject.toml
example/
  test.py
  data/
    uchuu/
      uchuu_catalog.h5        # lightweight example file supplied separately
py/
  fit_calibration.py          # offline NYU-VAGC calibration only
  mockgen/
    __init__.py
    __main__.py
    calibration.py
    cli.py
    config.py
    data.py
    hod.py
    lightcone.py
    models.py
    output.py
    pipeline.py
    properties.py
    data/
      nyu_vagc_calibration.npz
```

Runtime code does not depend on the repository directory name.

## Host halo input schema

The fixed input datasets are:

```text
halo_mass
halo_conc
halo_x
halo_y
halo_z
halo_vx
halo_vy
halo_vz
halo_vrms
```

There is no halo-delta/environment input.

## Offline NYU-VAGC calibration

`all0.dat` is used only to construct the bundled calibration table. Run this once with the full observational sample:

```bash
python py/fit_calibration.py /path/to/all0.dat
```

By default it writes:

```text
py/mockgen/data/nyu_vagc_calibration.npz
```

The table contains:

```text
mass_edges      fixed stellar-mass bin edges
color_params    [a1, mu1, sigma1, mu2, sigma2] for every mass bin
mr_coeffs       [c0, c_Ms, c_gr]
mr_sigma        residual scatter of the Mr relation
```

The fitted luminosity relation is

```text
Mr = c0 + c_Ms * log10(Mstar) + c_gr * (g-r) + Gaussian(0, mr_sigma)
```

The observational selections and double-Gaussian fitting logic follow the supplied reference implementation. To make calibration independent of each generated mock, the original dynamically defined high-mass bin edges are replaced by a fixed grid from `logM*=8` to `12.5`, while retaining the original low-mass edges `7, 7.4, 7.7, 8`.

## Install

```bash
pip install -e .
```

## End-to-end example/test

Place the lightweight halo file at:

```text
example/data/uchuu/uchuu_catalog.h5
```

and make sure the committed calibration table exists at:

```text
py/mockgen/data/nyu_vagc_calibration.npz
```

Then run:

```bash
python example/test.py
```

A successful run writes:

```text
example/test_mock.h5
```

That complete mock generation is the test; there is no separate test suite.

## Python API

### Center + area

```python
from mockgen import MockConfig, SkyRegion, generate_mock

sky = SkyRegion.centered_area(
    center_ra=150.0,
    center_dec=2.0,
    area_deg2=300.0,
)

config = MockConfig(
    uchuu_path="/path/to/uchuu_catalog.h5",
    output_path="mock.h5",
    sky=sky,
    z_min=0.0,
    z_max=1.0,
    random_seed=24,
    n_jobs=-1,
)

generate_mock(config)
```

`area_deg2` is treated as the exact area of a spherical cap centered on the requested RA/Dec. The cap angular radius is obtained from

```text
A = 2 pi (1 - cos theta)
```

with `A` converted from square degrees to steradians. Each candidate galaxy is retained when its great-circle angular separation from the requested center is no larger than `theta`.

### RA/Dec rectangle

```python
sky = SkyRegion.rectangle(
    ra_min=140.0,
    ra_max=160.0,
    dec_min=-5.0,
    dec_max=10.0,
)
```

A galaxy is retained when its Dec lies between the two Dec limits and its normalized RA lies inside the requested RA interval. RA wrap-around is supported. For example,

```python
SkyRegion.rectangle(350.0, 10.0, -5.0, 5.0)
```

means `RA >= 350 deg OR RA <= 10 deg`.

For both sky modes, an enclosing spherical cap is used only to prune periodic box replicas before expensive per-object calculations. The final angular selection is always performed with the exact requested circle or rectangle.

## CLI

Centered 300 deg2 region:

```bash
make-mock \
  --uchuu /path/to/uchuu_catalog.h5 \
  --output mock.h5 \
  --centered-area \
  --center-ra 150 \
  --center-dec 2 \
  --area 300 \
  --z-max 1.0
```

Rectangular region:

```bash
make-mock \
  --uchuu /path/to/uchuu_catalog.h5 \
  --output mock.h5 \
  --rectangle \
  --ra-min 140 \
  --ra-max 160 \
  --dec-min -5 \
  --dec-max 10 \
  --z-max 1.0
```

## Output schema

### `/galaxies`

```text
idx
index
type
ra
dec
x
y
z
vx
vy
vz
z_obs
z_com
Mh
Ms
Mr
mr
gr
```

`type=1` is central and `type=0` is satellite. `index` is the row index of the source host halo in the input halo catalog. `Ms` is log10 stellar mass and `Mh` is linear host halo mass.

### `/halos`

```text
idx
index
ra
dec
x
y
z
vx
vy
vz
z_obs
z_com
Mh
conc
vrms
```

`/halos[i]` is always the host of `/galaxies[i]`, in the same periodic replica.
