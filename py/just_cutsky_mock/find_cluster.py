import h5py


def find_clusters(galaxy_type, halo_mass, mass_cut):
    return (galaxy_type == 1) & (halo_mass >= mass_cut)


def read_clusters(path, mass_cut):
    with h5py.File(path, "r") as f:
        cluster_mask = find_clusters(f["galaxies/GAL_IS_CENTRAL"][:], f["halos/HALO_M200m"][:], mass_cut)
        clusters = {name: f["halos"][name][:][cluster_mask] for name in f["halos"]}
    return clusters
