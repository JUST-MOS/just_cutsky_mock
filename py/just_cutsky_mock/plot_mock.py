import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def plot_sky(galaxy_ra, galaxy_dec, cluster_ra, cluster_dec, footprint_mask, title="", galaxy_size=1, cluster_size=20):
    footprint = np.full(len(footprint_mask), hp.UNSEEN)
    footprint[footprint_mask] = 1.0
    hp.mollview(footprint, min=0.0, max=1.0, cmap="Greys", cbar=False, title=title)
    hp.graticule()
    hp.projscatter(galaxy_ra, galaxy_dec, lonlat=True, s=galaxy_size, alpha=0.5, label="Galaxies")
    hp.projscatter(cluster_ra, cluster_dec, lonlat=True, s=cluster_size, marker="o", label="Clusters")
    plt.legend()
    plt.show()
