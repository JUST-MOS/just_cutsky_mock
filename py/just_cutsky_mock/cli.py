from __future__ import annotations

import argparse

from .config import MockConfig, SkyRegion
from .pipeline import generate_mock


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a HEALPix cutsky galaxy + corresponding-host-halo mock.")
    p.add_argument("--uchuu", required=True, help="Path to host halo uchuu_catalog.h5")
    p.add_argument("--output", default="mock.h5", help="Output HDF5 path")
    p.add_argument("--z-min", type=float, default=0.0)
    p.add_argument("--z-max", type=float, default=2.4)
    p.add_argument("--nside", type=int, default=128, help="HEALPix NSIDE; RING ordering is used")
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--hod-chunk-size", type=int, default=500_000)

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--centered-area", action="store_true", help="Use center RA/Dec + area in deg^2")
    mode.add_argument("--rectangle", action="store_true", help="Use RA/Dec rectangular limits")

    p.add_argument("--center-ra", type=float)
    p.add_argument("--center-dec", type=float)
    p.add_argument("--area", type=float, help="Requested analytic area in deg^2 before HEALPix pixelization")
    p.add_argument("--ra-min", type=float)
    p.add_argument("--ra-max", type=float)
    p.add_argument("--dec-min", type=float)
    p.add_argument("--dec-max", type=float)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.centered_area:
        if args.center_ra is None or args.center_dec is None or args.area is None:
            raise SystemExit("--centered-area requires --center-ra --center-dec --area")
        sky = SkyRegion.centered_area(args.center_ra, args.center_dec, args.area)
    else:
        required = [args.ra_min, args.ra_max, args.dec_min, args.dec_max]
        if any(v is None for v in required):
            raise SystemExit("--rectangle requires --ra-min --ra-max --dec-min --dec-max")
        sky = SkyRegion.rectangle(args.ra_min, args.ra_max, args.dec_min, args.dec_max)

    config = MockConfig(
        uchuu_path=args.uchuu,
        output_path=args.output,
        sky=sky,
        z_min=args.z_min,
        z_max=args.z_max,
        healpix_nside=args.nside,
        random_seed=args.seed,
        n_jobs=args.n_jobs,
        hod_chunk_size=args.hod_chunk_size,
    )
    generate_mock(config)


if __name__ == "__main__":
    main()
