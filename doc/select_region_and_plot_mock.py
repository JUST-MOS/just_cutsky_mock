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
        mask = geometry.contains(f["galaxies/ra"][:], f["galaxies/dec"][:])
        galaxies = {name: f["galaxies"][name][:][mask] for name in f["galaxies"]}
        halos = {name: f["halos"][name][:][mask] for name in f["halos"]}
    return pixels, galaxies, halos


def find_clusters(galaxies, halos, mass_cut):
    mask = (galaxies["type"] == 1) & (halos["Mh"] >= mass_cut)
    clusters = {name: halos[name][mask] for name in halos}
    return clusters


def plot_mock(galaxies, clusters, pixels, nside, title=""):
    footprint = np.full(hp.nside2npix(nside), hp.UNSEEN)
    footprint[pixels] = 1.0
    hp.mollview(footprint, min=0.0, max=1.0, cmap="Greys", cbar=False, title=title)
    hp.graticule()
    hp.projscatter(galaxies["ra"], galaxies["dec"], lonlat=True, s=1, alpha=0.5, label="Galaxies")
    hp.projscatter(clusters["ra"], clusters["dec"], lonlat=True, s=20, marker="o", label="Clusters")
    plt.legend()
    plt.show()


mock_path = "mock.h5"
nside = 128
mass_cut = 1e14

circle_sky = SkyRegion.centered_area(center_ra=150.0, center_dec=2.0, area_deg2=300.0)
rectangle_sky = SkyRegion.rectangle(ra_min=140.0, ra_max=160.0, dec_min=-5.0, dec_max=5.0)

sky = circle_sky

pixels, galaxies, halos = select_mock(mock_path, sky, nside)
clusters = find_clusters(galaxies, halos, mass_cut)
plot_mock(galaxies, clusters, pixels, nside)
