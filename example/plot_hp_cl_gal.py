# The aim of this script is to:
# 1. select clusters from the galaxy mock generated with "generate_mock.py", which is saved in "example/uchuu_mini_mock.h5"
# 2. plot the clusters, galaxies in the mock and the healpix mask describing this galaxy mock

import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from just_cutsky_mock.find_cluster import read_clusters
from just_cutsky_mock.define_healpix import read_healpix, make_healpix_mask

mock_path = "uchuu_mini_mock.h5"
mass_cut = 1e13

clusters = read_clusters(mock_path, mass_cut)
pixels, nside, ordering = read_healpix(mock_path)
footprint_mask = make_healpix_mask(pixels, nside)

with h5py.File(mock_path, "r") as f:
    galaxy_ra = f["galaxies/GAL_RA"][:]
    galaxy_dec = f["galaxies/GAL_DEC"][:]

def plot_hp(footprint_mask, title="Healpix Mask of Galaxy Mock"):
    footprint = np.full(len(footprint_mask), hp.UNSEEN)
    footprint[footprint_mask] = 1.0
    hp.mollview(footprint, min=0.0, max=1.0, cmap="Greys", cbar=False, title=title)
    hp.graticule()
    plt.show()

def plot_gal(galaxy_ra, galaxy_dec, cluster_ra, cluster_dec, footprint_mask, title=fr"All Galaxies and Clusters with $\log M_h > {np.round(np.log10(mass_cut), 0)}$", galaxy_size=1, cluster_size=10):
    footprint = np.full(len(footprint_mask), hp.UNSEEN)
    footprint[footprint_mask] = 1.0
    hp.mollview(footprint, min=0.0, max=1.0, cmap="Greys", cbar=False, title=title)
    hp.graticule()
    hp.projscatter(galaxy_ra, galaxy_dec, lonlat=True, s=galaxy_size, alpha=0.5, label="Galaxies")
    hp.projscatter(cluster_ra, cluster_dec, lonlat=True, s=cluster_size, marker="o", label="Clusters")
    plt.legend()
    plt.show()

plot_hp(footprint_mask, title="Healpix Mask of Galaxy Mock")
plot_gal(galaxy_ra, galaxy_dec, clusters["HALO_RA"], clusters["HALO_DEC"], footprint_mask)
