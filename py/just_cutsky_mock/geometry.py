from __future__ import annotations

from dataclasses import dataclass

import healpy as hp
import numpy as np

from .config import SkyRegion


@dataclass(frozen=True)
class HealpixGeometry:
    """HEALPix representation of the requested angular footprint.

    The analytic input region (rectangle or center+area) is converted once to a
    list of RING-ordered HEALPix pixels. Pixels are selected by their centers.
    The lightcone then contains every galaxy whose angular position falls in
    one of those selected pixels.
    """

    region: SkyRegion
    nside: int
    pixels: np.ndarray
    nest: bool = False

    @classmethod
    def from_sky_region(cls, region: SkyRegion, nside: int) -> "HealpixGeometry":
        nside = int(nside)
        if not hp.isnsideok(nside, nest=False):
            raise ValueError(f"Invalid HEALPix nside={nside}.")

        if region.mode == "circle":
            theta = np.radians(90.0 - region.center_dec)
            phi = np.radians(region.center_ra)
            vec = hp.ang2vec(theta, phi)
            pixels = hp.query_disc(
                nside,
                vec,
                np.radians(region.angular_radius_deg),
                inclusive=False,
                nest=False,
            )
        else:
            # First select the Declination strip by pixel centers, then apply
            # the requested RA interval to those pixel centers. This preserves
            # the requested constant-RA / constant-Dec rectangle definition.
            theta_min = np.radians(90.0 - region.dec_max)
            theta_max = np.radians(90.0 - region.dec_min)
            strip = hp.query_strip(
                nside,
                theta_min,
                theta_max,
                inclusive=False,
                nest=False,
            )
            if region.ra_min == 0.0 and region.ra_max == 360.0:
                pixels = strip
            else:
                theta, phi = hp.pix2ang(nside, strip, nest=False)
                ra = np.degrees(phi) % 360.0
                if region.ra_min <= region.ra_max:
                    keep = (ra >= region.ra_min) & (ra <= region.ra_max)
                else:
                    keep = (ra >= region.ra_min) | (ra <= region.ra_max)
                pixels = strip[keep]

        pixels = np.unique(np.asarray(pixels, dtype=np.int64))
        return cls(region=region, nside=nside, pixels=pixels, nest=False)

    @property
    def ordering(self) -> str:
        return "RING"

    @property
    def pixel_area_deg2(self) -> float:
        return float(hp.nside2pixarea(self.nside, degrees=True))

    @property
    def area_deg2(self) -> float:
        return float(len(self.pixels) * self.pixel_area_deg2)

    @property
    def max_pixel_radius_deg(self) -> float:
        # Maximum center-to-corner angle of any pixel at this nside. Used only
        # to make periodic-tile pruning conservative at the footprint boundary.
        return float(np.degrees(hp.max_pixrad(self.nside)))

    def contains(self, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
        ra = np.asarray(ra_deg, dtype=float) % 360.0
        dec = np.asarray(dec_deg, dtype=float)
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        pix = hp.ang2pix(self.nside, theta, phi, nest=False)

        # pixels is sorted and unique, so searchsorted avoids allocating a
        # full-sky boolean lookup table at large nside.
        loc = np.searchsorted(self.pixels, pix)
        valid = loc < len(self.pixels)
        out = np.zeros(len(pix), dtype=bool)
        out[valid] = self.pixels[loc[valid]] == pix[valid]
        return out

    def bounding_cap(self) -> tuple[float, float, float]:
        """Conservative cap enclosing the selected HEALPix pixel union."""
        if len(self.pixels) == 0:
            ra, dec, _ = self.region.bounding_cap()
            return ra, dec, 0.0

        x, y, z = hp.pix2vec(self.nside, self.pixels, nest=False)
        vectors = np.column_stack((x, y, z))
        mean_vec = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm < 1.0e-12:
            return 0.0, 0.0, 180.0

        center = mean_vec / norm
        dots = np.clip(vectors @ center, -1.0, 1.0)
        radius = float(np.degrees(np.max(np.arccos(dots)))) + self.max_pixel_radius_deg
        if radius >= 180.0:
            return 0.0, 0.0, 180.0

        ra = float(np.degrees(np.arctan2(center[1], center[0])) % 360.0)
        dec = float(np.degrees(np.arcsin(np.clip(center[2], -1.0, 1.0))))
        return ra, dec, radius
