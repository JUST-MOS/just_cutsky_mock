from __future__ import annotations

import numpy as np

from .calibration import GalaxyCalibration
from .data import GalaxyBox
from .models import fred_cen, fred_sat


def sample_color(params, n_samples: int, rng: np.random.Generator, mode: str = "all"):
    a1, mu1, s1, mu2, s2 = params
    if mode == "red":
        return rng.normal(mu2, s2, n_samples)
    if mode == "blue":
        return rng.normal(mu1, s1, n_samples)
    choose_blue = rng.random(n_samples) < a1
    return np.where(choose_blue, rng.normal(mu1, s1, n_samples), rng.normal(mu2, s2, n_samples))


def rank_match(target, driver, rho: float):
    target = np.asarray(target)
    driver = np.asarray(driver)
    n = len(target)
    if n <= 1:
        return target.copy()

    target_std = np.std(target)
    driver_std = np.std(driver)
    if target_std == 0.0 or driver_std == 0.0:
        return target.copy()

    mixed = (
        rho * (driver - np.mean(driver)) / driver_std
        + np.sqrt(1.0 - rho**2) * (target - np.mean(target)) / target_std
    )
    return np.sort(target)[np.argsort(np.argsort(mixed))]


def assign_galaxy_properties(
    galaxies: GalaxyBox,
    calibration: GalaxyCalibration,
    *,
    random_seed: int,
) -> GalaxyBox:
    """Assign halo quenching, g-r color, and Mr from the bundled calibration."""
    if galaxies.size == 0:
        galaxies.gr = np.empty(0)
        galaxies.Mr = np.empty(0)
        return galaxies

    rng = np.random.default_rng(np.random.SeedSequence([random_seed, 2001]))

    cen_mask = galaxies.type == 1
    sat_mask = ~cen_mask
    quench = np.empty(galaxies.size, dtype=bool)
    quench[cen_mask] = rng.random(np.sum(cen_mask)) < fred_cen(galaxies.Mh[cen_mask])
    quench[sat_mask] = rng.random(np.sum(sat_mask)) < fred_sat(galaxies.Mh[sat_mask])

    mock_gr = np.full(galaxies.size, np.nan, dtype=float)
    max_idx = calibration.n_bins - 1
    bin_indices = np.clip(np.digitize(galaxies.Ms, calibration.mass_edges) - 1, 0, max_idx)

    for i in range(calibration.n_bins):
        idx_bin = np.flatnonzero(bin_indices == i)
        if idx_bin.size == 0:
            continue

        params = calibration.color_params[i]
        q_bin = quench[idx_bin]
        type_bin = galaxies.type[idx_bin]

        red_local = np.flatnonzero(q_bin)
        blue_local = np.flatnonzero(~q_bin)
        if red_local.size:
            mock_gr[idx_bin[red_local]] = sample_color(params, red_local.size, rng, mode="red")
        if blue_local.size:
            mock_gr[idx_bin[blue_local]] = sample_color(params, blue_local.size, rng, mode="blue")

        # Preserve the original halo-mass rank correlations.
        for state, rho_cen, rho_sat in ((q_bin, 0.5, 0.3), (~q_bin, 0.5, 0.3)):
            for is_central, rho in ((True, rho_cen), (False, rho_sat)):
                local = np.flatnonzero(state & (type_bin == int(is_central)))
                if local.size:
                    global_idx = idx_bin[local]
                    mock_gr[global_idx] = rank_match(
                        mock_gr[global_idx],
                        np.log10(galaxies.Mh[global_idx]),
                        rho,
                    )

    c0, c_ms, c_gr = calibration.mr_coeffs
    mock_Mr = c0 + c_ms * galaxies.Ms + c_gr * mock_gr
    mock_Mr += rng.normal(0.0, calibration.mr_sigma, galaxies.size)

    galaxies.gr = mock_gr
    galaxies.Mr = mock_Mr
    return galaxies
