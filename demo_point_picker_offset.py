#!/usr/bin/env python
"""Demo: register two offset 3D images using the point picker.

Same as demo_point_picker.py but adds a large translation offset to the
moving image.  Tests the translate-aware coordinate system: images render
at their true positions without allocating a huge zero-padded bounding box.
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
# The offset does not have to be positive in general, but for this demo my
# math assumes it is
true_offset = np.random.randint(5, 40, size=3)
initial_offset = true_offset + np.random.randint(-5, 5, size=3)

image1 = np.zeros(shape)
image2 = np.zeros(shape)

blob_centers_img1 = []
blob_centers_img2 = []

for _ in range(N):
    pad = 3 * stddev
    oz, oy, ox = true_offset
    z = np.random.randint(pad, oz + shape[0] - pad) 
    y = np.random.randint(pad, oy + shape[1] - pad)
    x = np.random.randint(pad, ox + shape[2] - pad)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]

    rr2 = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2
    spot = np.exp(-rr2 / (2 * stddev**2))
    image1 += np.random.uniform(0.5, 1.0) * spot
    blob_centers_img1.append((z, y, x))

    # Convert the spot coordinate into the shifted frame
    z, y, x = z - oz, y - oy, x - ox
    rr2 = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2
    spot = np.exp(-rr2 / (2 * stddev**2))
    image2 += np.random.uniform(0.5, 1.0) * spot
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

# ── Translation offset ────────────────────────────────────────────────────
# Simulate a coarse misalignment: layer2 is offset in world space.
# Use a random but reproducible offset so each run is testable.
rng = np.random.default_rng(42)
offset = rng.integers(-80, 80, size=3)
print(f"Translation offset applied to Moving layer: {offset}")

# ── Launch viewer & point picker ──────────────────────────────────────────
viewer = napari.Viewer()
viewer.add_image(image1, name="Fixed", colormap="green", blending="additive")
viewer.add_image(
    image2, name="Moving", colormap="magenta", blending="additive",
    translate=initial_offset,
)

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

# Pre-populate a few known correspondences so you can test immediately.
# These are in each layer's *data* coordinate space (indices into the array),
# which load_point_pairs will convert to world coordinates internally.
n_seed = min(5, N)
seed_pairs = {
    "Fixed": blob_centers_img1[:n_seed],
    "Moving": blob_centers_img2[:n_seed],
}

from qtpy.QtCore import QTimer
QTimer.singleShot(100, lambda: manager.load_registration_points(seed_pairs))

napari.run()
