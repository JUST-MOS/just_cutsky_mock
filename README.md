# cutsky_mock

Generate one specific product: a cutsky galaxy lightcone together with the corresponding host-halo lightcone, using the fixed HOD / quenching / color / luminosity-evolution logic in this repository.

The final product is a single HDF5 file:

```text
mock.h5
├── /galaxies
└── /halos
```

Row `i` in `/halos` is the host halo of row `i` in `/galaxies`. A halo is therefore intentionally repeated when several galaxies in the output share the same host.

## Input data

Only two external scientific inputs are required:

```text
uchuu_catalog.h5
all0.dat
```

The fixed Uchuu input schema is:

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

`all0.dat` uses the same NYU-VAGC column convention as the reference scripts: redshift in column 3, `Mr` in column 4, `g-r` in column 7, and log stellar mass in column 8.

There is no dependency on an Uchuu delta/environment file.

## Install

```bash
pip install -e .
```

Dependencies are declared in `pyproject.toml`.

## End-to-end test / example

Place the lightweight example files at exactly:

```text
examples/data/uchuu/uchuu_catalog.h5
examples/data/nyu-vagc/all0.dat
```

Then run:

```bash
python test.py
```

A successful run writes `test_mock.h5`. This is the intended end-to-end test; there is no separate test suite.

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
    nyu_vagc_path="/path/to/all0.dat",
    output_path="mock.h5",
    sky=sky,
    z_min=0.0,
    z_max=1.0,
    random_seed=24,
    n_jobs=-1,
)

generate_mock(config)
```

`area_deg2` is interpreted as the area of a spherical cap centered on the requested RA/Dec.

### RA/Dec rectangle

```python
sky = SkyRegion.rectangle(
    ra_min=140.0,
    ra_max=160.0,
    dec_min=-5.0,
    dec_max=10.0,
)
```

RA wrap-around is supported, e.g. `ra_min=350`, `ra_max=10`.

## CLI

Centered 300 deg² region:

```bash
make-mock \
  --uchuu /path/to/uchuu_catalog.h5 \
  --nyu-vagc /path/to/all0.dat \
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
  --nyu-vagc /path/to/all0.dat \
  --output mock.h5 \
  --rectangle \
  --ra-min 140 \
  --ra-max 160 \
  --dec-min -5 \
  --dec-max 10 \
  --z-max 1.0
```

## Fixed physical model

The public interface does not expose the HOD or evolution parameters. The implementation fixes the values used by the supplied reference code, including:

- `H0 = 67.74`
- `Om0 = 0.3089`
- `h = 0.6774`
- periodic box size `2000 Mpc/h`
- stellar-mass HOD threshold `1e8`
- the supplied Behroozi-style SHMR/scatter prescription
- the supplied satellite occupation and NFW placement prescription
- the supplied central/satellite halo-quenching laws used for color assignment
- the supplied double-Gaussian `g-r` model and halo-mass rank matching
- the supplied `Mr` regression
- the supplied RSD, apparent-magnitude, luminosity-evolution, and density-evolution prescriptions

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

`type=1` is central and `type=0` is satellite. `index` is the row index of the source halo in `uchuu_catalog.h5`. `Ms` is log10 stellar mass; `Mh` is linear halo mass, matching the galaxy-side convention in the reference pipeline.

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

The halo position/redshift is calculated for the same periodic replica that contains its corresponding galaxy. `/halos[i]` is always the host of `/galaxies[i]`.

## Repository layout

```text
pyproject.toml
README.md
test.py
examples/
  data/
    uchuu/
    nyu-vagc/
src/
  mockgen/
    __init__.py
    __main__.py
    cli.py
    config.py
    data.py
    hod.py
    lightcone.py
    models.py
    output.py
    pipeline.py
    properties.py
```

Runtime code never depends on the repository directory name or on the current working directory. Input/output paths are supplied explicitly.
