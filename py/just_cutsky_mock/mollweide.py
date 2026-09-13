#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp


def _read_attr_str(attrs, key, default=None):
    if key not in attrs:
        return default
    val = attrs[key]
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, np.bytes_):
        return val.decode("utf-8")
    return str(val)


def load_mock(mock_path, nside_override=None):
    with h5py.File(mock_path, "r") as f:
        if "geometry" not in f:
            raise KeyError("No /geometry group in mock.h5")
        if "pixels" not in f["geometry"]:
            raise KeyError("No /geometry/pixels dataset in mock.h5")
        if "galaxies" not in f:
            raise KeyError("No /galaxies group in mock.h5")
        if "ra" not in f["galaxies"] or "dec" not in f["galaxies"]:
            raise KeyError("No /galaxies/ra or /galaxies/dec in mock.h5")

        pixels = f["geometry"]["pixels"][:].astype(np.int64)

        if nside_override is not None:
            nside = int(nside_override)
        else:
            if "nside" not in f["geometry"].attrs:
                raise KeyError("No /geometry.attrs['nside']")
            nside = int(f["geometry"].attrs["nside"])

        ordering = _read_attr_str(f["geometry"].attrs, "ordering", default="RING").upper()
        nest = (ordering == "NEST")

        ra = f["galaxies"]["ra"][:].astype(np.float64)
        dec = f["galaxies"]["dec"][:].astype(np.float64)

    return pixels, nside, nest, ra, dec


def build_pixel_mask_map(pixels, nside):
    npix = hp.nside2npix(nside)
    m = np.full(npix, hp.UNSEEN, dtype=np.float64)
    m[pixels] = 1.0
    return m


def build_galaxy_count_map(ra, dec, nside, nest=False, restrict_pixels=None):
    """
    根据 galaxy RA/Dec 生成每个 HEALPix pixel 的星系数目。
    如果 restrict_pixels 不为 None，则只保留这些 pixels 内的计数，其余设为 UNSEEN。
    """
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)

    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    npix = hp.nside2npix(nside)

    counts = np.bincount(pix, minlength=npix).astype(np.float64)

    if restrict_pixels is not None:
        mask = np.zeros(npix, dtype=bool)
        mask[restrict_pixels] = True
        counts[~mask] = hp.UNSEEN
    else:
        counts[counts == 0] = hp.UNSEEN

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="return mollweide"
    )
    parser.add_argument(
        "mock_path",
        type=str,
        help="input mock path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="output path"
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=None,
        help="nside of healpix"
    )

    args = parser.parse_args()

    pixels, nside, nest, ra, dec = load_mock(args.mock_path, nside_override=args.nside)

    pixel_map = build_pixel_mask_map(pixels, nside)
    galaxy_map = build_galaxy_count_map(ra, dec, nside, nest=nest, restrict_pixels=pixels)

    galaxy_map_plot = galaxy_map.copy()
    valid = galaxy_map_plot != hp.UNSEEN
    galaxy_map_plot[valid] = np.log10(1.0 + galaxy_map_plot[valid])

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.mock_path))[0]
        args.output = os.path.join(os.path.dirname(args.mock_path), f"{base}_mollweide.png")

    fig = plt.figure(figsize=(14, 6))

    hp.mollview(
        pixel_map,
        fig=fig.number,
        sub=(1, 2, 1),
        title="Selected HEALPix Pixels",
        cmap="Blues",
        cbar=False,
        notext=False,
        nest=nest
    )

    hp.mollview(
        galaxy_map_plot,
        fig=fig.number,
        sub=(1, 2, 2),
        title=r"Galaxy Distribution: $\log_{10}(1+N_{\rm gal})$",
        cmap="viridis",
        cbar=True,
        unit=r"$\log_{10}(1+N_{\rm gal})$",
        notext=False,
        nest=nest
    )

    if args.title is not None:
        fig.suptitle(args.title, fontsize=14, y=0.98)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
