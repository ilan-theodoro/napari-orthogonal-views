#!/usr/bin/env python
"""Load two Zarr views into napari and wire up the point picker for registration.

Channel 0 is used as the fixed view and channel 2 as the moving view (Y-flipped)
by default. Pass ``--zarr-path`` to load both views from the same OME-Zarr, or
``--fixed-zarr-path`` / ``--moving-zarr-path`` to pull the two views from
different arrays (e.g. when aligning separate cameras).
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import dask.array as da
import napari
import numpy as np

# Allow running from a source checkout without installing the plugin first.
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from napari_orthogonal_views import (  # noqa: E402
    estimate_affine_from_points_components,
    estimate_affine_from_points_masked,
    show_point_picker,
)

def _channel_label(zarr_path: Path, channel: int) -> str:
    attrs_path = zarr_path.parent / ".zattrs"
    if not attrs_path.exists():
        return f"channel {channel}"

    with attrs_path.open() as file:
        attrs = json.load(file)

    channels = attrs.get("omero", {}).get("channels", [])
    if channel >= len(channels):
        return f"channel {channel}"

    return channels[channel].get("label") or f"channel {channel}"


def _slice_channel(
    data: da.Array,
    *,
    timepoint: int,
    channel: int,
    flip_y: bool = False,
    flip_x: bool = False,
) -> da.Array:
    if data.ndim != 5:
        raise ValueError(
            "Expected a 5D Zarr array with axes (T, C, Z, Y, X), "
            f"but got shape {data.shape}."
        )
    if not 0 <= timepoint < data.shape[0]:
        raise ValueError(
            f"timepoint {timepoint} is out of bounds for T={data.shape[0]}"
        )
    if not 0 <= channel < data.shape[1]:
        raise ValueError(
            f"channel {channel} is out of bounds for C={data.shape[1]}"
        )

    volume = data[timepoint, channel]
    if flip_y:
        volume = volume[:, ::-1, :]
    if flip_x:
        volume = volume[:, :, ::-1]
    return volume.compute()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open one or two Zarr arrays in napari with the point picker "
            "wired up for affine registration."
        )
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=None,
        help=(
            "Path to a 5D Zarr array used as the source for any view whose "
            "own --fixed-zarr-path / --moving-zarr-path override is not "
            "supplied. No default -- you must pass at least one of these "
            "three options."
        ),
    )
    parser.add_argument(
        "--fixed-zarr-path",
        type=Path,
        default=None,
        help="Override the Zarr array used for the fixed view.",
    )
    parser.add_argument(
        "--moving-zarr-path",
        type=Path,
        default=None,
        help="Override the Zarr array used for the moving view.",
    )
    parser.add_argument(
        "--timepoint",
        type=int,
        default=0,
        help="Timepoint index to load.",
    )
    parser.add_argument(
        "--fixed-channel",
        type=int,
        default=0,
        help="Channel index used as the fixed/reference view.",
    )
    parser.add_argument(
        "--moving-channel",
        type=int,
        default=2,
        help="Channel index used as the moving view.",
    )
    parser.add_argument(
        "--no-flip-moving-y",
        action="store_true",
        help="Disable the default vertical Y-axis flip on the moving view.",
    )
    parser.add_argument(
        "--flip-moving-x",
        action="store_true",
        help=(
            "Flip the moving view horizontally along X. Useful when aligning "
            "views from different cameras."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shared_path = (
        args.zarr_path.expanduser() if args.zarr_path is not None else None
    )
    fixed_zarr_path = (
        args.fixed_zarr_path.expanduser()
        if args.fixed_zarr_path is not None
        else shared_path
    )
    moving_zarr_path = (
        args.moving_zarr_path.expanduser()
        if args.moving_zarr_path is not None
        else shared_path
    )
    if fixed_zarr_path is None or moving_zarr_path is None:
        raise SystemExit(
            "No Zarr path provided. Pass --zarr-path for a shared source, or "
            "--fixed-zarr-path and --moving-zarr-path to load each view from "
            "its own array."
        )

    fixed_data = da.from_zarr(str(fixed_zarr_path))
    moving_data = (
        fixed_data
        if moving_zarr_path == fixed_zarr_path
        else da.from_zarr(str(moving_zarr_path))
    )

    flip_moving_y = not args.no_flip_moving_y
    flip_moving_x = args.flip_moving_x

    fixed = _slice_channel(
        fixed_data,
        timepoint=args.timepoint,
        channel=args.fixed_channel,
    )
    moving = _slice_channel(
        moving_data,
        timepoint=args.timepoint,
        channel=args.moving_channel,
        flip_y=flip_moving_y,
        flip_x=flip_moving_x,
    )

    fixed_name = (
        f"View {args.fixed_channel}: "
        f"{_channel_label(fixed_zarr_path, args.fixed_channel)}"
    )
    moving_name = (
        f"View {args.moving_channel}: "
        f"{_channel_label(moving_zarr_path, args.moving_channel)}"
    )
    flip_tags = []
    if flip_moving_y:
        flip_tags.append("Y flipped")
    if flip_moving_x:
        flip_tags.append("X flipped")
    if flip_tags:
        moving_name += f" ({', '.join(flip_tags)})"
    moving_name += " (moving view)"

    print(f"Fixed zarr:  {fixed_zarr_path}")
    print(f"Moving zarr: {moving_zarr_path}")
    print(f"Fixed shape (T, C, Z, Y, X):  {fixed_data.shape}")
    print(f"Moving shape (T, C, Z, Y, X): {moving_data.shape}")
    print(
        f"Loaded timepoint {args.timepoint}, "
        f"fixed channel {args.fixed_channel}, "
        f"moving channel {args.moving_channel}"
    )
    if flip_tags:
        print(f"Moving view flips: {', '.join(flip_tags)}")

    viewer = napari.Viewer(title="Zarr view registration")
    viewer.add_image(
        fixed,
        name=fixed_name,
        colormap="green",
        blending="additive",
        contrast_limits=(0, 65535),
        metadata={
            "source_zarr": str(fixed_zarr_path),
            "timepoint": args.timepoint,
            "channel": args.fixed_channel,
            "y_flipped": False,
            "x_flipped": False,
        },
    )
    viewer.add_image(
        moving,
        name=moving_name,
        colormap="magenta",
        blending="additive",
        contrast_limits=(0, 65535),
        metadata={
            "source_zarr": str(moving_zarr_path),
            "timepoint": args.timepoint,
            "channel": args.moving_channel,
            "y_flipped": flip_moving_y,
            "x_flipped": flip_moving_x,
        },
    )

    viewer.dims.current_step = tuple(int(size // 2) for size in fixed.shape)

    # NOTE: Use this estimator when aligning two views acquired by the same
    # camera (e.g. different stains on a shared detector). It disallows
    # rotations that depend on Z -- the slices should keep their geometry
    # regardless of their Z position -- and locks the scale to prevent drift.
    # affine_estimator = partial(
    #     estimate_affine_from_points_components,
    #     fix_rotation=(False, True, True),
    #     fix_scale=True,
    # )

    # NOTE: This estimator restricts the affine to the XY slice plane only:
    # the mask leaves the Z row as identity and zeroes the Z column, so only
    # in-plane rotation/scale/shear and the YX translations can move. Use it
    # when aligning two physical cameras from raw (non-processed) views, where
    # the cameras share the Z stage geometry and only the XY relationship
    # needs to be solved.
    mask = np.zeros((3, 4), dtype=bool)
    mask[1:3, 1:4] = True
    affine_estimator = partial(
        estimate_affine_from_points_masked,
        mask=mask,
    )

    show_point_picker(
        viewer,
        layer1_name=fixed_name,
        layer2_name=moving_name,
        affine_estimator=affine_estimator,
    )
    napari.run()


if __name__ == "__main__":
    main()
