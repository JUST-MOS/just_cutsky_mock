from __future__ import annotations

from dataclasses import dataclass
from math import acos, pi
from pathlib import Path
from typing import Literal

import numpy as np


H0 = 67.74
OM0 = 0.3089
H = 0.6774
BOX_SIZE = 2000.0
MS_THRESHOLD = 1.0e8
HDF5_COMPRESSION = "lzf"


@dataclass(frozen=True)
class SkyRegion:
    """Analytic sky region used to choose HEALPix pixels.

    Two modes are supported:
      * ``rectangle``: RA/Dec limits in degrees, including RA wrap-around.
      * ``circle``: a spherical cap centered at (RA, Dec), with area in deg^2.

    This analytic region does not directly cut galaxies. It first determines
    the HEALPix pixel set; the final catalog is the union of those pixels.
    """

    mode: Literal["rectangle", "circle"]
    ra_min: float | None = None
    ra_max: float | None = None
    dec_min: float | None = None
    dec_max: float | None = None
    center_ra: float | None = None
    center_dec: float | None = None
    area_deg2: float | None = None

    @classmethod
    def rectangle(cls, ra_min: float, ra_max: float, dec_min: float, dec_max: float) -> "SkyRegion":
        if not (-90.0 <= dec_min < dec_max <= 90.0):
            raise ValueError("Require -90 <= dec_min < dec_max <= 90 degrees.")
        raw_min = float(ra_min)
        raw_max = float(ra_max)
        if abs(raw_max - raw_min) >= 360.0:
            norm_min, norm_max = 0.0, 360.0
        else:
            norm_min, norm_max = raw_min % 360.0, raw_max % 360.0
        return cls(
            mode="rectangle",
            ra_min=norm_min,
            ra_max=norm_max,
            dec_min=float(dec_min),
            dec_max=float(dec_max),
        )

    @classmethod
    def centered_area(cls, center_ra: float, center_dec: float, area_deg2: float) -> "SkyRegion":
        if not (-90.0 <= center_dec <= 90.0):
            raise ValueError("center_dec must be in [-90, 90] degrees.")
        if not (0.0 < area_deg2 <= 4.0 * pi * (180.0 / pi) ** 2):
            raise ValueError("area_deg2 must be positive and no larger than the full sky.")
        return cls(
            mode="circle",
            center_ra=float(center_ra) % 360.0,
            center_dec=float(center_dec),
            area_deg2=float(area_deg2),
        )

    @property
    def angular_radius_deg(self) -> float:
        if self.mode != "circle":
            raise AttributeError("angular_radius_deg is only defined for circle mode.")
        area_sr = self.area_deg2 * (pi / 180.0) ** 2
        return float(np.degrees(acos(np.clip(1.0 - area_sr / (2.0 * pi), -1.0, 1.0))))

    def bounding_cap(self) -> tuple[float, float, float]:
        """Return an enclosing spherical cap as (RA, Dec, radius) in degrees."""
        if self.mode == "circle":
            return self.center_ra, self.center_dec, self.angular_radius_deg

        if self.ra_min == 0.0 and self.ra_max == 360.0:
            return 0.0, 0.0, 180.0
        if self.ra_min <= self.ra_max:
            width = self.ra_max - self.ra_min
        else:
            width = self.ra_max + 360.0 - self.ra_min
        center_ra = (self.ra_min + 0.5 * width) % 360.0
        center_dec = 0.5 * (self.dec_min + self.dec_max)

        if width > 180.0:
            return center_ra, center_dec, 180.0

        corners_ra = np.array([self.ra_min, self.ra_min, self.ra_max, self.ra_max])
        corners_dec = np.array([self.dec_min, self.dec_max, self.dec_min, self.dec_max])
        radius = float(np.max(_angular_separation_deg(center_ra, center_dec, corners_ra, corners_dec)))
        return center_ra, center_dec, min(radius, 180.0)

    def describe(self) -> str:
        if self.mode == "circle":
            return (
                f"circle(center_ra={self.center_ra:.6f}, center_dec={self.center_dec:.6f}, "
                f"area_deg2={self.area_deg2:.6f}, radius_deg={self.angular_radius_deg:.6f})"
            )
        return (
            f"rectangle(ra_min={self.ra_min:.6f}, ra_max={self.ra_max:.6f}, "
            f"dec_min={self.dec_min:.6f}, dec_max={self.dec_max:.6f})"
        )


def _angular_separation_deg(ra1, dec1, ra2, dec2):
    ra1 = np.radians(ra1)
    dec1 = np.radians(dec1)
    ra2 = np.radians(ra2)
    dec2 = np.radians(dec2)
    cosang = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


@dataclass(frozen=True)
class MockConfig:
    halo_path: Path | str
    output_path: Path | str
    sky: SkyRegion
    z_min: float = 0.0
    z_max: float = 2.4
    healpix_nside: int = 128
    random_seed: int = 24
    n_jobs: int = -1
    hod_chunk_size: int = 500_000

    def __post_init__(self):
        object.__setattr__(self, "halo_path", Path(self.halo_path).expanduser().resolve())
        object.__setattr__(self, "output_path", Path(self.output_path).expanduser().resolve())
        if not (0.0 <= self.z_min < self.z_max):
            raise ValueError("Require 0 <= z_min < z_max.")
        if self.healpix_nside <= 0:
            raise ValueError("healpix_nside must be positive.")
        if self.hod_chunk_size <= 0:
            raise ValueError("hod_chunk_size must be positive.")
