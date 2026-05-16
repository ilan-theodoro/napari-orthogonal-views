#!/usr/bin/env python
"""Demo: register two 3D images using the point picker.

Creates two synthetic volumes (Gaussian blobs), applies a known shear to the
second, then launches the orthogonal-view point picker so you can align them.
"""

import sys
from pathlib import Path

import napari
import numpy as np
from scipy.ndimage import affine_transform

# Add src to path so we can import the plugin without installing
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from napari_orthogonal_views import estimate_affine_from_points_no_scale, show_point_picker

# ── Synthetic data ────────────────────────────────────────────────────────
stddev = 1
N = 100
shape = (200, 200, 200)

image1 = np.zeros(shape)
image2 = np.zeros(shape)

blob_centers_img1 = []
blob_centers_img2 = []

for _ in range(N):
    z = np.random.randint(3 * stddev, shape[0] - 3 * stddev)
    y = np.random.randint(3 * stddev, shape[1] - 3 * stddev)
    x = np.random.randint(3 * stddev, shape[2] - 3 * stddev)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    rr2 = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2

    spot = np.exp(-rr2 / (2 * stddev**2))
    image1 += np.random.uniform(0.5, 1.0) * spot
    image2 += np.random.uniform(0.5, 1.0) * spot

    blob_centers_img1.append((z, y, x))
    blob_centers_img2.append((z, y, x))

# Apply a small shear to image2
shear_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.02, 1.0, 0.05],
    [0.0, 0.05, 1.0],
])
image2 = affine_transform(image2, shear_matrix, order=1)

# Transform blob centres to match the shear (inverse maps old→new position)
shear_inv = np.linalg.inv(shear_matrix)
blob_centers_img2 = [
    tuple(np.round(shear_inv @ np.array(c)).astype(int))
    for c in blob_centers_img2
]

# ── Launch viewer & point picker ──────────────────────────────────────────
viewer = napari.Viewer()
viewer.add_image(image1, name="Fixed", colormap="green", blending="additive")
viewer.add_image(image2, name="Moving", colormap="magenta", blending="additive")

manager = show_point_picker(
    viewer, layer1_name="Fixed", layer2_name="Moving",
    affine_estimator=estimate_affine_from_points_no_scale,
)


def _on_affine_applied(affine):
    pairs = manager.get_registration_points()
    np.set_printoptions(suppress=True, precision=10, linewidth=200)
    print("=== Affine applied ===")
    print("Selected point pairs (data space):")
    for l1_pt, l2_pt in zip(pairs["Fixed"], pairs["Moving"]):
        print(f"  Fixed: {l1_pt}   Moving: {l2_pt}")
    print("World-space affine (Moving → Fixed):")
    print(affine)


manager.point_picker_widget.affine_applied.connect(_on_affine_applied)

# Pre-populate a few known correspondences so you can test immediately
n_seed = min(5, N)
seed_pairs = {
    "Fixed": blob_centers_img1[:n_seed],
    "Moving": blob_centers_img2[:n_seed],
}

from qtpy.QtCore import QTimer
QTimer.singleShot(100, lambda: manager.load_registration_points(seed_pairs))

napari.run()
