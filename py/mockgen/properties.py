from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .data import GalaxyBox, ObservationalSample, vlim
from .models import fred_cen, fred_sat


def double_gaussian(x, a1, mu1, sigma1, mu2, sigma2):
    return (
        a1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) / (sigma1 * np.sqrt(2.0 * np.pi))
        + (1.0 - a1)
        * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
        / (sigma2 * np.sqrt(2.0 * np.pi))
    )


def fit_step(data, bins: int = 25):
    data = np.asarray(data)
    data = data[(data > 0.2) & (data < 1.2)]
    if data.size < 8:
        raise ValueError("Not enough NYU-VAGC g-r values to fit the double Gaussian.")

    counts, edges = np.histogram(data, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    popt, _ = curve_fit(
        double_gaussian,
        centers,
        counts,
        p0=[0.5, 0.6, 0.1, 0.9, 0.1],
        bounds=([0.0, 0.5, 0.1, 0.8, 0.05], [1.0, 0.9, 0.3, 1.2, 0.2]),
        maxfev=5000,
    )
    return popt


def _safe_fit_step(data, fallback):
    try:
        return fit_step(data)
    except (ValueError, RuntimeError):
        return fit_step(fallback)


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
    obs: ObservationalSample,
    *,
    random_seed: int,
) -> GalaxyBox:
    """Assign halo quenching, g-r color, and Mr using the supplied model."""
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
    mock_ms_min = 8.0
    mock_ms_max = float(np.max(galaxies.Ms))
    mass_edges = np.append(np.array([7.0, 7.4, 7.7]), np.linspace(mock_ms_min, mock_ms_max + 0.0001, 21))
    max_idx = len(mass_edges) - 2
    bin_indices = np.clip(np.digitize(galaxies.Ms, mass_edges) - 1, 0, max_idx)

    global_color_sample = obs.gr[(obs.gr > 0.2) & (obs.gr < 1.2)]
    if global_color_sample.size < 8:
        raise ValueError("NYU-VAGC example data does not contain enough valid g-r values.")

    for i in range(max_idx + 1):
        low = mass_edges[i]
        high = mass_edges[i + 1]
        idx_bin = np.flatnonzero(bin_indices == i)
        if idx_bin.size == 0:
            continue

        obs_mask = (obs.Ms > low) & (obs.Ms < high) & (obs.z < vlim(low))
        sample_gr = obs.gr[obs_mask]
        params = _safe_fit_step(sample_gr, global_color_sample)

        q_bin = quench[idx_bin]
        type_bin = galaxies.type[idx_bin]

        red_local = np.flatnonzero(q_bin)
        blue_local = np.flatnonzero(~q_bin)
        if red_local.size:
            mock_gr[idx_bin[red_local]] = sample_color(params, red_local.size, rng, mode="red")
        if blue_local.size:
            mock_gr[idx_bin[blue_local]] = sample_color(params, blue_local.size, rng, mode="blue")

        for is_red, rho_cen, rho_sat in ((True, 0.5, 0.3), (False, 0.5, 0.3)):
            state = q_bin if is_red else ~q_bin
            for is_central, rho in ((True, rho_cen), (False, rho_sat)):
                local = np.flatnonzero(state & (type_bin == int(is_central)))
                if local.size:
                    global_idx = idx_bin[local]
                    mock_gr[global_idx] = rank_match(
                        mock_gr[global_idx],
                        np.log10(galaxies.Mh[global_idx]),
                        rho,
                    )

    if obs.Ms_mag.size < 3:
        raise ValueError(
            "NYU-VAGC example data does not contain enough objects in 0.08 < z < 0.12 "
            "for the Mr regression."
        )

    A_obs = np.column_stack((np.ones_like(obs.Ms_mag), obs.Ms_mag, obs.gr_mag))
    coeffs, _, _, _ = np.linalg.lstsq(A_obs, obs.Mr_mag, rcond=None)
    obs_Mr_pred = A_obs @ coeffs
    sigma_Mr = float(np.std(obs.Mr_mag - obs_Mr_pred))

    mock_Mr = np.full(galaxies.size, np.nan, dtype=float)
    valid = np.isfinite(mock_gr)
    A_mock = np.column_stack((np.ones(np.sum(valid)), galaxies.Ms[valid], mock_gr[valid]))
    mock_Mr[valid] = A_mock @ coeffs + rng.normal(0.0, sigma_Mr, np.sum(valid))

    galaxies.gr = mock_gr
    galaxies.Mr = mock_Mr
    return galaxies
