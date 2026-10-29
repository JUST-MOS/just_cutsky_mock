# The aim of this script is to 

import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from just_cutsky_mock.config import SkyRegion
from just_cutsky_mock.geometry import HealpixGeometry


def select_mock(path, sky, nside):
    # define healpix pixels for the targeted sky region
    geometry = HealpixGeometry.from_sky_region(sky, nside)
    pixels = geometry.pixels
    with h5py.File(path, "r") as f:
        mask = geometry.contains(f["GAL_RA"][:], f["GAL_DEC"][:])
        halo_mask = (f["GAL_IS_CENTRAL"][:] == 1)
        galaxies = {name: f[name][:][mask] for name in ["GAL_RA", "GAL_DEC", "GAL_IS_CENTRAL"]}
        halos = {name: f[name][:][(mask & halo_mask)] for name in f["GAL_RA", "GAL_DEC", "HALO_M200m"]}
    return pixels, galaxies, halos


def find_clusters(halos, mass_cut):
    mask = (halos["HALO_M200m"] >= mass_cut)
    clusters = {name: halos[name][mask] for name in halos}
    return clusters


def plot_mock(galaxies, clusters, pixels, nside, title=""):
    footprint = np.full(hp.nside2npix(nside), hp.UNSEEN)
    footprint[pixels] = 1.0
    hp.mollview(footprint, min=0.0, max=1.0, cmap="Greys", cbar=False, title=title)
    hp.graticule()
    hp.projscatter(galaxies["GAL_RA"], galaxies["GAL_DEC"], lonlat=True, s=1, alpha=0.5, label="Galaxies")
    hp.projscatter(clusters["GAL_RA"], clusters["GAL_DEC"], lonlat=True, s=20, marker="o", label="Clusters")
    plt.legend()
    plt.show()


mock_path = "/home/hjunyu/Just/Mock/mocks/mock.h5" # file path on Gravity
nside = 128
mass_cut = 1e13

# Define the targeted sky region (either a circle or a rectangle sky region)
circle_sky = SkyRegion.centered_area(center_ra=45.0, center_dec=45.0, area_deg2=10.0)
rectangle_sky = SkyRegion.rectangle(ra_min=44.0, ra_max=46.0, dec_min=44.0, dec_max=46.0)
sky = circle_sky or rectangle_sky

# get the healpix pixels, galaxies and host halos in the targeted sky region
pixels, galaxies, halos = select_mock(mock_path, sky, nside)
# get the clusters with host halo mass M200m > mass_cut
clusters = find_clusters(halos, mass_cut)
# plot the clusters/galaxies in the healpix footprint
plot_mock(galaxies, clusters, pixels, nside)
