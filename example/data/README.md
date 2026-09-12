Place the lightweight host halo catalog here:

```text
example/data/uchuu/uchuu_catalog.h5
```

The file must use the same dataset names as the production host halo catalog:

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

The NYU-VAGC calibration is not an example/runtime input. It is fitted once with
`py/fit_calibration.py` and stored inside `py/mockgen/data/nyu_vagc_calibration.npz`.
