from __future__ import annotations

import numpy as np
from scipy.special import erf


H_DEFAULT = 0.6774


def behroozi_shmr(Ms):
    Ms = np.asarray(Ms, dtype=float)
    zM1 = 10.0 ** 12.23 * 0.72
    zM0 = 10.0 ** 10.31
    zbeta = 0.34
    zdelta = 0.42 + 0.07
    zgamma = 1.21 - 0.4
    m = Ms / zM0
    out = np.zeros_like(m)
    mask = m > 0
    mm = m[mask]
    out[mask] = zM1 * mm**zbeta * 10.0 ** (mm**zdelta / (1.0 + mm ** (-zgamma)) - 0.5)
    return out


class FixedHODModel:
    """Fixed HOD/SHMR model matching the supplied reference implementation."""

    def __init__(self, ms_threshold: float = 1.0e8, h: float = H_DEFAULT):
        self.ms_threshold = float(ms_threshold)
        self.h = float(h)

        temp_ms = np.logspace(1.0, 12.5, 1000)
        temp_mh = behroozi_shmr(temp_ms)
        self.shmr_mh_grid = temp_mh
        self.shmr_logms_grid = np.log10(temp_ms)

        self.u_axis = np.concatenate(
            [
                np.linspace(0.0, 0.9, 100, endpoint=False),
                1.0 - np.logspace(-1.0, -12.0, 150),
            ]
        )
        self.u_axis[0] = 0.0
        self.u_axis[-1] = 1.0
        self.logmh_axis = np.linspace(10.0, 16.0, 200)
        self.logms_sat_table = self._build_satellite_mass_lut()

        self.s_grid = np.logspace(-4.0, 5.0, 10000)
        self.s_grid = np.insert(self.s_grid, 0, 0.0)
        self.f_grid = np.log(1.0 + self.s_grid) - self.s_grid / (1.0 + self.s_grid)

    def mean_stellar_mass(self, Mh):
        Mh = np.asarray(Mh, dtype=float)
        logms = np.interp(Mh, self.shmr_mh_grid, self.shmr_logms_grid)
        return 10.0**logms

    def shmr_scatter(self, Mh):
        Mh = np.asarray(Mh, dtype=float)
        zM1 = 10.0 ** 12.23 * 0.72
        zeta = -0.04
        val_const = 0.217147
        out = np.full_like(Mh, val_const, dtype=float)
        mask = Mh >= zM1
        out[mask] = (0.5 + zeta * np.log10(Mh[mask] / zM1)) * 2.0 * val_const
        return out

    def p_cen(self, Ms, Mh):
        mean = np.log10(self.mean_stellar_mass(Mh))
        scatter = self.shmr_scatter(Mh)
        x = np.log10(np.asarray(Ms, dtype=float))
        return 1.0 - 0.5 * (1.0 + erf((x - mean) / (scatter * np.sqrt(2.0))))

    def sat_characteristic_mass(self, Ms):
        zBsat = 8.98
        zbetasat = 0.90 - 0.2
        factor = 1.0e12 / self.h
        return factor * zBsat * (behroozi_shmr(Ms) / factor) ** zbetasat

    def sat_cutoff_mass(self, Ms):
        zBcut = 0.86
        zbetacut = 0.41
        factor = 1.0e12 / self.h
        return factor * zBcut * (behroozi_shmr(Ms) / factor) ** zbetacut

    def num_sat_above_threshold(self, Ms, Mh):
        alphasat = 1.0 - 0.2
        ncen = self.p_cen(Ms, Mh)
        msat = self.sat_characteristic_mass(Ms)
        mcut = self.sat_cutoff_mass(Ms)
        return ncen * (Mh / msat) ** alphasat * np.exp(-mcut / Mh)

    def expected_num_sat(self, Mh):
        Mh = np.asarray(Mh, dtype=float)
        return self.num_sat_above_threshold(np.full_like(Mh, self.ms_threshold), Mh)

    def _build_satellite_mass_lut(self):
        ms_integ = np.logspace(np.log10(self.ms_threshold), 13.0, 2000)
        log_ms = np.log10(ms_integ)
        table = np.empty((len(self.logmh_axis), len(self.u_axis)), dtype=float)

        for i, logmh in enumerate(self.logmh_axis):
            mh = 10.0**logmh
            mh_arr = np.full_like(ms_integ, mh)
            n_cumulative = self.num_sat_above_threshold(ms_integ, mh_arr)
            n_total = n_cumulative[0]
            if n_total < 1.0e-10:
                table[i, :] = np.log10(self.ms_threshold)
                continue

            cdf = 1.0 - n_cumulative / n_total
            cdf[0] = 0.0
            unique_cdf, unique_idx = np.unique(cdf, return_index=True)
            unique_cdf[-1] = 1.0
            table[i, :] = np.interp(self.u_axis, unique_cdf, log_ms[unique_idx])
        return table

    def sample_satellite_masses(self, parent_mh, u):
        parent_mh = np.asarray(parent_mh, dtype=float)
        u = np.asarray(u, dtype=float)
        logmh = np.log10(parent_mh)

        i = np.searchsorted(self.logmh_axis, logmh) - 1
        j = np.searchsorted(self.u_axis, u) - 1
        i = np.clip(i, 0, len(self.logmh_axis) - 2)
        j = np.clip(j, 0, len(self.u_axis) - 2)

        x1 = self.logmh_axis[i]
        x2 = self.logmh_axis[i + 1]
        y1 = self.u_axis[j]
        y2 = self.u_axis[j + 1]
        q11 = self.logms_sat_table[i, j]
        q21 = self.logms_sat_table[i + 1, j]
        q12 = self.logms_sat_table[i, j + 1]
        q22 = self.logms_sat_table[i + 1, j + 1]

        tx = np.divide(logmh - x1, x2 - x1, out=np.zeros_like(logmh), where=(x2 != x1))
        ty = np.divide(u - y1, y2 - y1, out=np.zeros_like(u), where=(y2 != y1))
        logms = (
            q11 * (1.0 - tx) * (1.0 - ty)
            + q21 * tx * (1.0 - ty)
            + q12 * (1.0 - tx) * ty
            + q22 * tx * ty
        )
        return 10.0**logms

    def sample_nfw_positions(self, parent_mh, parent_conc, parent_pos, u_r, u_phi, u_costheta, box_size):
        parent_mh = np.asarray(parent_mh, dtype=float)
        conc = np.asarray(parent_conc, dtype=float) * 0.86
        parent_pos = np.asarray(parent_pos, dtype=float)

        r200 = (3.0 * parent_mh / (800.0 * np.pi * 0.3089 * 2.77e11)) ** (1.0 / 3.0)
        fc = np.log(1.0 + conc) - conc / (1.0 + conc)
        y = np.asarray(u_r) * fc
        s = np.interp(y, self.f_grid, self.s_grid)
        r = (s / conc) * r200

        phi = 2.0 * np.pi * np.asarray(u_phi)
        costheta = 2.0 * np.asarray(u_costheta) - 1.0
        sintheta = np.sqrt(np.maximum(0.0, 1.0 - costheta**2))
        offset = np.column_stack(
            (
                r * sintheta * np.cos(phi),
                r * sintheta * np.sin(phi),
                r * costheta,
            )
        )
        return np.mod(parent_pos + offset, box_size)


def fred_cen(Mh):
    Mh = np.asarray(Mh, dtype=float)
    return 1.0 - np.exp(-(Mh / (10.0**11.93)) ** 0.39)


def fred_sat(Mh):
    Mh = np.asarray(Mh, dtype=float)
    return 1.0 - np.exp(-(Mh / (10.0**12.17)) ** 0.15)


def delta_mag_function(z):
    O, P, Q = 3.0843, 0.9105, -0.1337
    del O, P
    return -2.5 * Q * np.log10(1.0 + np.asarray(z))


def weight_function(z):
    O, P, Q = 3.0843, 0.9105, -0.1337
    del Q
    return np.exp(-((np.asarray(z) / P) ** O))
