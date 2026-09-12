# JUST CUTSKY MOCK (v1.0)

Generating a galaxy light-cone for a requested sky region with the corresponding host-halo light-cone.

Resulting in a single HDF5 file:

```text
mock.h5
├── /galaxies
└── /halos
└── /geometry
```


## Host halo input structure

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
example/data/uchuu/uchuu_catalog_mini.h5
```

Then run:

```bash
python example/test.py
```

Which will result in:

```text
example/test_mock.h5
```


## Output structure

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

Each row in `/halos` corresponds to the host halo of the same row in `/galaxies`.

### `/geometry`

```text
pixels
```

The selected RING-ordered HEALPix pixel numbers.
