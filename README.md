# JUST CUTSKY MOCK (v1.0)

Generating a galaxy light-cone for a requested sky region with the corresponding host-halo light-cone.

Resulting in a single HDF5 file:

```text
mock.h5
├── /galaxies
└── /halos
```

Every row in `/halos` matches the same row in `/galaxies`. A host halo is repeated when several output galaxies share it.


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


## Install

```bash
pip install -e .
```


## Example

The mini halo catalog file:

```text
example/data/uchuu/uchuu_catalog.h5
```

Then run:

```bash
python example/test.py
```

Which will result in:

```text
example/test_mock.h5
```


## Python API

### Center + area

```python
from just_cutsky_mock import MockConfig, SkyRegion, generate_mock

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

### RA/Dec rectangle

```python
sky = SkyRegion.rectangle(
    ra_min=140.0,
    ra_max=160.0,
    dec_min=-5.0,
    dec_max=10.0,
)
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
