import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass

import affiners
import numpy as np
from napari.utils.notifications import show_warning
from napari.viewer import Viewer
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass
class PointPair:
    """Store a matched pair of coordinates for registration."""

    pair_id: int
    layer1_coords: tuple | None = None
    layer2_coords: tuple | None = None


class ShowUpdateWidget(QWidget):
    """Widget with Show and Update buttons for a single coordinate."""

    show_clicked = Signal()
    update_clicked = Signal()

    def __init__(self):
        super().__init__()

        self.show_button = QPushButton("Show")
        self.update_button = QPushButton("Update")

        self.show_button.clicked.connect(self.show_clicked.emit)
        self.update_button.clicked.connect(self.update_clicked.emit)

        layout = QHBoxLayout()
        layout.addWidget(self.show_button)
        layout.addWidget(self.update_button)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)


def _validate_point_arrays(
    source_points: np.ndarray, target_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce point arrays for affine estimation."""
    source_points = np.asarray(source_points)
    target_points = np.asarray(target_points)

    if source_points.ndim != 2 or target_points.ndim != 2:
        raise ValueError("Point arrays must be 2-dimensional (N, ndim).")
    if source_points.shape != target_points.shape:
        raise ValueError(
            "Source and target point arrays must have the same shape."
        )

    n, ndim = source_points.shape
    if n < ndim + 1:
        raise ValueError(
            f"Need at least {ndim + 1} point pairs for {ndim}D affine estimation, "
            f"got {n}."
        )
    return source_points, target_points


def estimate_affine_from_points(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
    """Estimate a full affine transform (with scaling) from matched point pairs.

    Parameters
    ----------
    source_points : np.ndarray
        (N, ndim) array of source coordinates.
    target_points : np.ndarray
        (N, ndim) array of target coordinates.

    Returns
    -------
    np.ndarray
        (ndim+1, ndim+1) homogeneous affine matrix.

    Raises
    ------
    ValueError
        If fewer than ndim+1 point pairs are provided.
    """
    source_points, target_points = _validate_point_arrays(
        source_points, target_points
    )
    n, ndim = source_points.shape

    # Add homogeneous coordinate
    src_homo = np.hstack([source_points, np.ones((n, 1))])

    # Solve least squares: tgt = src_homo @ T.T
    T, _, _, _ = np.linalg.lstsq(src_homo, target_points, rcond=None)

    # Build full homogeneous matrix
    affine_matrix = np.eye(ndim + 1)
    affine_matrix[:ndim, :] = T.T

    return affine_matrix


def estimate_affine_from_points_no_scale(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
    """Estimate an affine transform without scaling (diagonal of linear part fixed to 1).

    Solves constrained least squares column-by-column. For each target dimension j,
    the coefficient T[j,j] (the scale along axis j) is fixed to 1 and the remaining
    coefficients (off-diagonal shear/rotation + translation) are optimized.

    Parameters
    ----------
    source_points : np.ndarray
        (N, ndim) array of source coordinates.
    target_points : np.ndarray
        (N, ndim) array of target coordinates.

    Returns
    -------
    np.ndarray
        (ndim+1, ndim+1) homogeneous affine matrix with ones on the linear diagonal.

    Raises
    ------
    ValueError
        If fewer than ndim+1 point pairs are provided.
    """
    source_points, target_points = _validate_point_arrays(
        source_points, target_points
    )
    n, ndim = source_points.shape

    src_homo = np.hstack([source_points, np.ones((n, 1))])
    affine_matrix = np.eye(ndim + 1)

    for j in range(ndim):
        # Move the known diagonal contribution (1 * source_j) to the RHS
        rhs = target_points[:, j] - source_points[:, j]

        # Solve for all coefficients except the diagonal one
        other_cols = [i for i in range(ndim + 1) if i != j]
        lhs = src_homo[:, other_cols]

        coeffs, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)

        # affine_matrix[j, j] is already 1 from np.eye
        for k, col_idx in enumerate(other_cols):
            if col_idx < ndim:
                affine_matrix[j, col_idx] = coeffs[k]
            else:
                affine_matrix[j, ndim] = coeffs[k]  # translation

    return affine_matrix


def estimate_affine_from_points_masked(
    source_points: np.ndarray,
    target_points: np.ndarray,
    mask: np.ndarray,
    fixed_values: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate an affine with a user-specified subset of parameters free.

    Generalizes :func:`estimate_affine_from_points_no_scale`: instead of
    hard-coding which entries of the linear+translation block are fixed,
    the caller passes a boolean mask the same shape as that block.

    For each target dimension ``j``, the entries of row ``j`` of the affine
    flagged as free in ``mask`` are optimized via least squares; the rest
    are held to ``fixed_values[j]`` (default: identity — ones on the
    diagonal, zeros elsewhere).

    Parameters
    ----------
    source_points : np.ndarray
        (N, ndim) array of source coordinates.
    target_points : np.ndarray
        (N, ndim) array of target coordinates.
    mask : array_like of bool
        (ndim, ndim+1) mask. ``True`` (or any truthy value) marks an entry
        as free to optimize; ``False`` holds it at ``fixed_values``.
    fixed_values : np.ndarray, optional
        (ndim, ndim+1) values used for entries where ``mask`` is False.
        Defaults to ``np.eye(ndim, ndim+1)`` so non-optimized entries take
        identity values.

    Returns
    -------
    np.ndarray
        (ndim+1, ndim+1) homogeneous affine matrix.

    Raises
    ------
    ValueError
        If point arrays are inconsistent, ``mask``/``fixed_values`` shapes
        don't match ``(ndim, ndim+1)``, or any row of ``mask`` has more
        free parameters than data points.
    """
    source_points, target_points = _validate_point_arrays(
        source_points, target_points
    )
    n, ndim = source_points.shape

    mask = np.asarray(mask).astype(bool)
    expected_shape = (ndim, ndim + 1)
    if mask.shape != expected_shape:
        raise ValueError(
            f"mask must have shape {expected_shape}, got {mask.shape}."
        )

    if fixed_values is None:
        fixed_values = np.eye(ndim, ndim + 1)
    else:
        fixed_values = np.asarray(fixed_values, dtype=float)
        if fixed_values.shape != expected_shape:
            raise ValueError(
                f"fixed_values must have shape {expected_shape}, "
                f"got {fixed_values.shape}."
            )

    src_homo = np.hstack([source_points, np.ones((n, 1))])
    affine_matrix = np.eye(ndim + 1)
    affine_matrix[:ndim, :] = fixed_values

    for j in range(ndim):
        free_idx = np.flatnonzero(mask[j])
        fixed_idx = np.flatnonzero(~mask[j])

        if free_idx.size == 0:
            continue
        if free_idx.size > n:
            raise ValueError(
                f"Row {j} has {free_idx.size} free parameters but only {n} "
                "point pairs are available."
            )

        # Subtract the contribution of the held-fixed entries from the RHS,
        # then solve least squares for the free ones.
        rhs = (
            target_points[:, j]
            - src_homo[:, fixed_idx] @ fixed_values[j, fixed_idx]
        )
        lhs = src_homo[:, free_idx]
        coeffs, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
        affine_matrix[j, free_idx] = coeffs

    return affine_matrix


def _decompose_linear_3d(
    L: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor a 3x3 linear map L as ``R · diag(s) · H``.

    R is a rotation (so(3), returned as an axis-angle 3-vector), ``s`` are
    the three scales (positive when ``det(L) > 0``), and H is unit
    upper-triangular shear, returned as ``(h01, h02, h12)``.

    Used to seed :func:`estimate_affine_from_points_components` from the
    unconstrained least-squares estimate.
    """
    from scipy.spatial.transform import Rotation

    Q, K = np.linalg.qr(L)
    # Force positive diagonals on K by absorbing signs into Q
    signs = np.sign(np.diag(K))
    signs[signs == 0] = 1
    K = signs[:, None] * K
    Q = Q * signs[None, :]
    # Resolve reflections so Q is a proper rotation (det = +1)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
        K[-1, :] *= -1

    scale = np.diag(K).copy()
    H = K / scale[:, None]
    shear = np.array([H[0, 1], H[0, 2], H[1, 2]])
    rotvec = Rotation.from_matrix(Q).as_rotvec()
    return rotvec, scale, shear


def _compose_affine_3d(
    rotvec: np.ndarray,
    scale: np.ndarray,
    shear: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Inverse of :func:`_decompose_linear_3d`; builds the 4x4 affine."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_rotvec(rotvec).as_matrix()
    S = np.diag(scale)
    H = np.array(
        [
            [1.0, shear[0], shear[1]],
            [0.0, 1.0, shear[2]],
            [0.0, 0.0, 1.0],
        ]
    )
    A = np.eye(4)
    A[:3, :3] = R @ S @ H
    A[:3, 3] = translation
    return A


def _normalize_component_mask(mask, name: str) -> np.ndarray:
    """Coerce a ``fix_*`` argument to a length-3 boolean array.

    A scalar bool broadcasts across all three axes; a length-3 sequence is
    converted as-is.
    """
    if isinstance(mask, (bool, np.bool_)):
        return np.array([bool(mask)] * 3, dtype=bool)
    arr = np.asarray(mask)
    if arr.shape != (3,):
        raise ValueError(
            f"{name} must be a bool or a length-3 sequence of bools, "
            f"got shape {arr.shape}."
        )
    return arr.astype(bool)


def estimate_affine_from_points_components(
    source_points: np.ndarray,
    target_points: np.ndarray,
    fix_scale=False,
    fix_shear=False,
    fix_rotation=False,
    fix_translation=False,
) -> np.ndarray:
    """Estimate a 3D affine with selected per-axis components held fixed.

    The affine is parameterized as

        A = T · R · S · H

    with 12 free parameters total:

    - rotation: so(3) axis-angle ``(rx, ry, rz)``
    - scale:    diagonal ``(sx, sy, sz)``
    - shear:    unit-upper-triangular ``(h01, h02, h12)``
    - translation: ``(tx, ty, tz)``

    Each ``fix_<component>`` argument controls which entries of that
    component are held at the identity value during optimization. It can be:

    - a single bool — broadcast across all three axes;
    - a length-3 sequence of bools — per-axis (``True`` = fixed).

    Example: ``fix_rotation=(False, True, True)`` keeps ``rx`` free and
    pins ``ry``/``rz`` to 0.

    Identity values used for held entries are: scale=1, shear=0,
    rotation=0 (axis-angle), translation=0. Free entries are initialized
    from the QR decomposition of the unconstrained linear LS estimate,
    then refined via :func:`scipy.optimize.least_squares` (LM).

    Only 3D point sets are supported.

    Returns
    -------
    np.ndarray
        4x4 homogeneous affine matrix.
    """
    from scipy.optimize import least_squares

    source_points, target_points = _validate_point_arrays(
        source_points, target_points
    )
    _, ndim = source_points.shape
    if ndim != 3:
        raise ValueError(
            "estimate_affine_from_points_components supports only 3D points; "
            f"got ndim={ndim}."
        )

    fix_rot = _normalize_component_mask(fix_rotation, "fix_rotation")
    fix_sca = _normalize_component_mask(fix_scale, "fix_scale")
    fix_she = _normalize_component_mask(fix_shear, "fix_shear")
    fix_tra = _normalize_component_mask(fix_translation, "fix_translation")
    free_mask = np.concatenate([~fix_rot, ~fix_sca, ~fix_she, ~fix_tra])

    id_full = np.concatenate(
        [np.zeros(3), np.ones(3), np.zeros(3), np.zeros(3)]
    )

    # Seed free entries from an unconstrained LS fit. If everything is
    # fixed there's nothing to optimize — return identity directly.
    if not free_mask.any():
        return np.eye(4)

    A_init = estimate_affine_from_points(source_points, target_points)
    try:
        init_rotvec, init_scale, init_shear = _decompose_linear_3d(
            A_init[:3, :3]
        )
        init_trans = A_init[:3, 3]
    except np.linalg.LinAlgError:
        init_rotvec = np.zeros(3)
        init_scale = np.ones(3)
        init_shear = np.zeros(3)
        init_trans = np.zeros(3)

    init_full = np.concatenate(
        [init_rotvec, init_scale, init_shear, init_trans]
    )
    x0 = init_full[free_mask]

    def unpack(x: np.ndarray) -> tuple:
        full = id_full.copy()
        full[free_mask] = x
        return full[0:3], full[3:6], full[6:9], full[9:12]

    def residuals(x: np.ndarray) -> np.ndarray:
        rotvec, scale, shear, trans = unpack(x)
        A = _compose_affine_3d(rotvec, scale, shear, trans)
        pred = (A[:3, :3] @ source_points.T).T + A[:3, 3]
        return (pred - target_points).ravel()

    result = least_squares(residuals, x0, method="lm")
    rotvec, scale, shear, trans = unpack(result.x)
    return _compose_affine_3d(rotvec, scale, shear, trans)


class PointPickerWidget(QWidget):
    """Widget for picking matched point pairs across two image layers."""

    affine_applied = Signal(np.ndarray)

    def __init__(
        self,
        viewer: Viewer,
        layer1_name: str = "Image 1",
        layer2_name: str = "Image 2",
        affine_estimator: (
            Callable[[np.ndarray, np.ndarray], np.ndarray] | None
        ) = None,
    ):
        super().__init__()

        self.viewer = viewer
        self.layer1_name = layer1_name
        self.layer2_name = layer2_name
        self.affine_estimator = affine_estimator or estimate_affine_from_points
        self.point_pairs: dict[int, PointPair] = (
            {}
        )  # Use dict for stable pair_id lookup
        self._next_pair_id = 0
        self.transform_snapshot: dict | None = None
        self._applied_affine: np.ndarray | None = None
        self._original_translates: dict[str, np.ndarray] = {}
        self._translates_captured = False
        self._nudge_offset = np.zeros(3)
        self._affine_base_translate: np.ndarray | None = None

        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Pair", layer1_name, layer2_name, ""]
        )
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 80)
        self.table.setColumnWidth(1, 150)
        self.table.setColumnWidth(2, 150)
        self.table.setColumnWidth(3, 30)

        # Add new pair button
        self.add_button = QPushButton("Add new pair")
        self.add_button.clicked.connect(self.add_pair)

        # Save / Load buttons for point pairs
        self.save_points_button = QPushButton("Save Points")
        self.save_points_button.clicked.connect(self._save_points_dialog)
        self.load_points_button = QPushButton("Load Points")
        self.load_points_button.clicked.connect(self._load_points_dialog)
        save_load_layout = QHBoxLayout()
        save_load_layout.addWidget(self.save_points_button)
        save_load_layout.addWidget(self.load_points_button)

        # Transform buttons
        self.apply_button = QPushButton("Apply Estimated Affine")
        self.apply_button.clicked.connect(self._apply_affine)
        self.apply_button.setEnabled(False)

        self.reset_button = QPushButton("Reset Transform")
        self.reset_button.clicked.connect(self._reset_transform)
        self.reset_button.setEnabled(False)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.reset_button)

        # Nudge offset controls
        nudge_group = QGroupBox("Nudge Offset")
        nudge_layout = QVBoxLayout()
        nudge_spin_layout = QHBoxLayout()
        self._nudge_spins = []
        for axis_label in ("Z", "Y", "X"):
            label = QLabel(axis_label)
            spin = QDoubleSpinBox()
            spin.setRange(-10000, 10000)
            spin.setSingleStep(1.0)
            spin.setDecimals(1)
            spin.setValue(0.0)
            spin.valueChanged.connect(self._apply_nudge)
            nudge_spin_layout.addWidget(label)
            nudge_spin_layout.addWidget(spin)
            self._nudge_spins.append(spin)
        nudge_layout.addLayout(nudge_spin_layout)
        self._reset_nudge_button = QPushButton("Reset Nudge")
        self._reset_nudge_button.clicked.connect(self._reset_nudge)
        nudge_layout.addWidget(self._reset_nudge_button)
        nudge_group.setLayout(nudge_layout)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.add_button)
        layout.addLayout(save_load_layout)
        layout.addWidget(nudge_group)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _capture_translates(self) -> None:
        """Read and store the original translate vectors from both layers.

        Called lazily the first time coordinates are needed, since layers may
        not exist at __init__ time.
        """
        if self._translates_captured:
            return
        for name in (self.layer1_name, self.layer2_name):
            if name in self.viewer.layers:
                self._original_translates[name] = np.array(
                    self.viewer.layers[name].translate, dtype=float
                )
            else:
                self._original_translates[name] = np.zeros(3)
        self._translates_captured = True

    def _transform_coords(self, coords: tuple, affine: np.ndarray) -> tuple:
        """Apply a homogeneous affine matrix to world coordinates."""
        ndim = affine.shape[0] - 1
        p = np.array(coords[:ndim], dtype=float)
        transformed = affine[:ndim, :ndim] @ p + affine[:ndim, ndim]
        result = list(coords)  # preserve any extra dims unchanged
        result[:ndim] = transformed.tolist()
        return tuple(result)

    def _get_base_translate(self) -> np.ndarray:
        """Return what layer2.translate would be without the nudge offset."""
        if self._affine_base_translate is not None:
            return self._affine_base_translate.copy()
        self._capture_translates()
        return self._original_translates.get(
            self.layer2_name, np.zeros(3)
        ).copy()

    def _update_layer2_translate(self) -> None:
        """Set layer2.translate to base + nudge_offset."""
        if self.layer2_name not in self.viewer.layers:
            return
        layer = self.viewer.layers[self.layer2_name]
        base = self._get_base_translate()
        layer.translate = base + self._nudge_offset

    def _apply_nudge(self) -> None:
        """Called when any nudge spinbox value changes."""
        self._capture_translates()
        self._nudge_offset = np.array(
            [spin.value() for spin in self._nudge_spins]
        )
        self._update_layer2_translate()

    def _reset_nudge(self) -> None:
        """Reset all nudge spinboxes to zero."""
        for spin in self._nudge_spins:
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self._nudge_offset = np.zeros(3)
        self._reset_nudge_button.setStyleSheet("")
        self._update_layer2_translate()

    def cleanup_nudge(self) -> None:
        """Reset the nudge offset and restore layer2.translate to its base value."""
        if not np.allclose(self._nudge_offset, 0):
            self._reset_nudge()

    def add_pair(self) -> None:
        """Add a new point pair row to the table."""

        pair_id = self._next_pair_id
        self._next_pair_id += 1

        point_pair = PointPair(pair_id=pair_id)
        self.point_pairs[pair_id] = point_pair

        row = self.table.rowCount()
        self.table.insertRow(row)

        # Column 0: Pair name
        pair_item = QTableWidgetItem(f"Pair {pair_id}")
        pair_item.setFlags(pair_item.flags() & ~Qt.ItemIsEditable)
        pair_item.setData(Qt.UserRole, pair_id)  # Store pair_id in the item
        self.table.setItem(row, 0, pair_item)

        # Column 1: fixed layer show/update widget
        layer1_widget = ShowUpdateWidget()
        layer1_widget.show_clicked.connect(
            lambda pid=pair_id: self._show_coordinate(pid, "layer1")
        )
        layer1_widget.update_clicked.connect(
            lambda pid=pair_id: self._update_coordinate(pid, "layer1")
        )
        self.table.setCellWidget(row, 1, layer1_widget)

        # Column 2: moving layer show/update widget
        layer2_widget = ShowUpdateWidget()
        layer2_widget.show_clicked.connect(
            lambda pid=pair_id: self._show_coordinate(pid, "layer2")
        )
        layer2_widget.update_clicked.connect(
            lambda pid=pair_id: self._update_coordinate(pid, "layer2")
        )
        self.table.setCellWidget(row, 2, layer2_widget)

        # Column 3: Delete button
        delete_button = QPushButton("×")
        delete_button.setMaximumWidth(30)
        delete_button.clicked.connect(
            lambda checked, pid=pair_id: self._delete_pair(pid)
        )
        self.table.setCellWidget(row, 3, delete_button)

        # Update button states
        self._update_button_states()

    def _show_coordinate(self, pair_id: int, layer: str) -> None:
        """Move crosshair to the saved world coordinate for the given pair and layer."""

        if pair_id not in self.point_pairs:
            return

        self._capture_translates()

        point_pair = self.point_pairs[pair_id]
        coords = (
            point_pair.layer1_coords
            if layer == "layer1"
            else point_pair.layer2_coords
        )

        if coords is None:
            # No coordinate saved yet, default to layer's origin in world space
            layer_name = (
                self.layer1_name if layer == "layer1" else self.layer2_name
            )
            translate = self._original_translates.get(layer_name, np.zeros(3))
            coords = tuple(translate.tolist())

        if layer == "layer2":
            if self._applied_affine is not None:
                coords = self._transform_coords(coords, self._applied_affine)
            if not np.allclose(self._nudge_offset, 0):
                coords = tuple(
                    (np.array(coords) + self._nudge_offset).tolist()
                )

        self.viewer.dims.point = coords

    def _update_coordinate(self, pair_id: int, layer: str) -> None:
        """Save the current crosshair world position for the given pair and layer."""

        if pair_id not in self.point_pairs:
            return

        self._capture_translates()

        current_coords = tuple(self.viewer.dims.point)

        if layer == "layer2":
            if not np.allclose(self._nudge_offset, 0):
                current_coords = tuple(
                    (np.array(current_coords) - self._nudge_offset).tolist()
                )
            if self._applied_affine is not None:
                inv = np.linalg.inv(self._applied_affine)
                current_coords = self._transform_coords(current_coords, inv)

        point_pair = self.point_pairs[pair_id]

        if layer == "layer1":
            point_pair.layer1_coords = current_coords
        else:
            point_pair.layer2_coords = current_coords

        # Update button states
        self._update_button_states()

    def _delete_pair(self, pair_id: int) -> None:
        """Delete the point pair with the given pair_id."""

        if pair_id not in self.point_pairs:
            return

        # Remove from data
        del self.point_pairs[pair_id]

        # Find and remove the row from the table
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.data(Qt.UserRole) == pair_id:
                self.table.removeRow(row)
                break

        # Update button states
        self._update_button_states()

    def _world_to_original_data(
        self, coords: tuple | None, layer_name: str
    ) -> tuple | None:
        """Convert world coordinates to the layer's original data coordinates.

        Subtracts the layer's original translate (captured before any padding
        or affine operations) so the returned value is an index into the
        original data array.
        """
        if coords is None:
            return None
        self._capture_translates()
        translate = self._original_translates.get(layer_name, np.zeros(3))
        arr = np.array(coords[: len(translate)], dtype=float)
        arr -= translate
        return tuple(arr.tolist())

    def _original_data_to_world(self, coords: tuple, layer_name: str) -> tuple:
        """Convert original data coordinates to world coordinates."""
        self._capture_translates()
        translate = self._original_translates.get(layer_name, np.zeros(3))
        arr = np.array(coords[: len(translate)], dtype=float)
        arr += translate
        return tuple(arr.tolist())

    def get_point_pairs(self) -> dict:
        """Return point pairs as two parallel coordinate lists.

        Coordinates are returned in each layer's original data space
        (before any padding or affine), matching the format expected by
        ``load_point_pairs()``.

        Returns:
            Dict keyed by layer name, each a list of (z, y, x) tuples.
            Only pairs where both coordinates are set are included.
        """
        layer1 = []
        layer2 = []
        for pair in self.point_pairs.values():
            if (
                pair.layer1_coords is not None
                and pair.layer2_coords is not None
            ):
                layer1.append(
                    self._world_to_original_data(
                        pair.layer1_coords, self.layer1_name
                    )
                )
                layer2.append(
                    self._world_to_original_data(
                        pair.layer2_coords, self.layer2_name
                    )
                )
        return {self.layer1_name: layer1, self.layer2_name: layer2}

    def clear_pairs(self) -> None:
        """Clear all point pairs."""

        self.point_pairs.clear()
        self.table.setRowCount(0)
        self._next_pair_id = 0

    def load_point_pairs(self, pairs: dict) -> None:
        """Load point pairs into the widget.

        Parameters
        ----------
        pairs : dict
            Dict keyed by layer name, each a list of (z, y, x) coordinate
            tuples in each layer's original data space.  Matches the format
            returned by ``get_point_pairs()``.
        """
        self._capture_translates()
        self.clear_pairs()
        for l1, l2 in zip(pairs[self.layer1_name], pairs[self.layer2_name]):
            self.add_pair()
            pair_id = self._next_pair_id - 1
            point_pair = self.point_pairs[pair_id]
            point_pair.layer1_coords = self._original_data_to_world(
                tuple(l1), self.layer1_name
            )
            point_pair.layer2_coords = self._original_data_to_world(
                tuple(l2), self.layer2_name
            )
        self._update_button_states()

    def _save_points_dialog(self) -> None:
        """Prompt for a path and write current point pairs as JSON."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save point pairs",
            "point_pairs.json",
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        pairs = self.get_point_pairs()
        payload = {
            "layer1_name": self.layer1_name,
            "layer2_name": self.layer2_name,
            "pairs": {
                self.layer1_name: [list(c) for c in pairs[self.layer1_name]],
                self.layer2_name: [list(c) for c in pairs[self.layer2_name]],
            },
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[PointPicker] Saved {len(pairs[self.layer1_name])} pairs to {path}")

    def _load_points_dialog(self) -> None:
        """Prompt for a JSON file and load point pairs from it."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load point pairs",
            "",
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        with open(path) as fh:
            payload = json.load(fh)
        # Accept both the wrapped format written by _save_points_dialog and
        # a bare dict matching the load_point_pairs() signature.
        if "pairs" in payload and isinstance(payload["pairs"], dict):
            pairs_raw = payload["pairs"]
            src_l1 = payload.get("layer1_name")
            src_l2 = payload.get("layer2_name")
        else:
            pairs_raw = payload
            src_l1 = src_l2 = None

        # Remap layer-name keys to the current widget's layer names if needed.
        if src_l1 and src_l2 and (
            src_l1 != self.layer1_name or src_l2 != self.layer2_name
        ):
            pairs = {
                self.layer1_name: pairs_raw[src_l1],
                self.layer2_name: pairs_raw[src_l2],
            }
        else:
            pairs = pairs_raw

        self.load_point_pairs(pairs)
        print(f"[PointPicker] Loaded {len(pairs[self.layer1_name])} pairs from {path}")

    def _update_button_states(self) -> None:
        """Update the enabled state of the Apply and Reset buttons."""

        # Get valid pairs (both coordinates set)
        valid_pairs = [
            p
            for p in self.point_pairs.values()
            if p.layer1_coords is not None and p.layer2_coords is not None
        ]

        # Enable Apply button if we have at least 4 valid pairs
        self.apply_button.setEnabled(len(valid_pairs) >= 4)

        # Enable Reset button if we have a snapshot
        self.reset_button.setEnabled(self.transform_snapshot is not None)

    def _estimate_affine_transform(self) -> np.ndarray | None:
        """Estimate affine transform from point pairs using least squares.

        Returns:
            Homogeneous affine matrix (4x4 for 3D, 3x3 for 2D) or None if insufficient pairs.
        """

        # Get valid pairs
        valid_pairs = [
            p
            for p in self.point_pairs.values()
            if p.layer1_coords is not None and p.layer2_coords is not None
        ]

        if len(valid_pairs) < 4:
            return None

        # Transform the moving layer to match the fixed layer:
        # - Source points are from the moving layer (layer2_coords)
        # - Target points are from the fixed layer (layer1_coords)
        src_pts = np.array([p.layer2_coords for p in valid_pairs])
        tgt_pts = np.array([p.layer1_coords for p in valid_pairs])

        affine = self.affine_estimator(src_pts, tgt_pts)

        return affine

    def _apply_affine(self) -> None:
        """Apply the estimated affine transform to the moving layer.

        Pre-transforms the image data using scipy rather than setting
        layer.affine, because napari does not fully support non-orthogonal
        slicing — off-diagonal affine components (shear/rotation) are
        stripped in non-displayed dimensions, causing orthoviews to render
        incorrectly.

        The output array is sized to fit the axis-aligned bounding box of
        layer2's transformed corners, so no image data is clipped.  The
        layer's translate is set to the bounding box origin so that the
        result sits at the correct world-space position.
        """

        # Check if target layer exists
        if self.layer2_name not in self.viewer.layers:
            return

        self._capture_translates()

        layer = self.viewer.layers[self.layer2_name]

        # Snapshot current state if not already done
        if self.transform_snapshot is None:
            self.transform_snapshot = {
                "affine": layer.affine.affine_matrix.copy(),
                "data": layer.data,
                "translate": np.array(layer.translate, dtype=float)
                - self._nudge_offset,
            }

        # Compute affine (world-space: layer2-world → layer1-world)
        affine = self._estimate_affine_transform()
        if affine is not None:
            pairs = self.get_point_pairs()
            np.set_printoptions(suppress=True, precision=10, linewidth=200)
            print("[PointPicker] Selected point pairs (original data space):")
            for l1_pt, l2_pt in zip(
                pairs[self.layer1_name], pairs[self.layer2_name]
            ):
                print(f"  {self.layer1_name}: {l1_pt}   {self.layer2_name}: {l2_pt}")
            print("[PointPicker] World-space affine (layer2 → layer1):")
            print(affine)
            data_affine = self.get_estimated_affine()
            print("[PointPicker] Data-space affine (layer2 → layer1):")
            print(data_affine)

            ndim = affine.shape[0] - 1
            A_lin = affine[:ndim, :ndim]
            A_trans = affine[:ndim, ndim]
            A_lin_inv = np.linalg.inv(A_lin)

            # Layer2's original translate (not current — current may be bbox_min
            # from a previous application)
            T2 = self.transform_snapshot["translate"][:ndim]
            data_shape = np.array(
                self.transform_snapshot["data"].shape[:ndim], dtype=float
            )

            # Build the 2^ndim corners of layer2 in world space
            corners_data = np.array(
                list(itertools.product(*[(0, s) for s in data_shape]))
            )
            corners_world = corners_data + T2

            # Transform corners through the affine (layer2-world → layer1-world)
            corners_transformed = (A_lin @ corners_world.T).T + A_trans

            # Axis-aligned bounding box of the transformed volume
            bbox_min = np.floor(corners_transformed.min(axis=0)).astype(int)
            bbox_max = np.ceil(corners_transformed.max(axis=0)).astype(int)
            output_shape = tuple((bbox_max - bbox_min).tolist())

            # Inverse mapping: output index o → world w = o + bbox_min
            # then world → layer2-data: d2 = A_lin_inv @ (w - A_trans) - T2
            # combined: d2 = A_lin_inv @ o + A_lin_inv @ (bbox_min - A_trans) - T2
            inv_offset = A_lin_inv @ (bbox_min.astype(float) - A_trans) - T2

            affiners_matrix = np.eye(ndim + 1)
            affiners_matrix[:ndim, :ndim] = A_lin_inv
            affiners_matrix[:ndim, ndim] = inv_offset
            import torch

            data = (
                torch.from_numpy(self.transform_snapshot["data"])
                .to(torch.float16)
                .numpy()
            )

            target_dtype = self.transform_snapshot["data"].dtype
            transformed = affiners.affine_transform(
                data,
                affiners_matrix,
                output_shape=output_shape,
            )
            layer.data = transformed

            # Restore original affine (identity in most cases) so no
            # non-orthogonal slicing occurs.
            layer.affine = self.transform_snapshot["affine"]

            # Place the output at the bounding box origin in world space,
            # preserving any active nudge offset.
            self._affine_base_translate = bbox_min.astype(float)
            layer.translate = self._affine_base_translate + self._nudge_offset

            self._applied_affine = affine
            self.affine_applied.emit(affine)
            self._update_button_states()

            if not np.allclose(self._nudge_offset, 0):
                self._reset_nudge_button.setStyleSheet(
                    "QPushButton { background-color: #c0392b; color: white; }"
                )
                show_warning(
                    "Affine applied while nudge is active. "
                    "Consider resetting the nudge offset to see the "
                    "aligned result without the extra translation."
                )

    def _reset_transform(self) -> None:
        """Reset the transform of the moving layer to the snapshot state."""

        if self.transform_snapshot is None:
            return

        if self.layer2_name not in self.viewer.layers:
            return

        layer = self.viewer.layers[self.layer2_name]
        layer.data = self.transform_snapshot["data"]
        layer.affine = self.transform_snapshot["affine"]
        layer.translate = (
            self.transform_snapshot["translate"] + self._nudge_offset
        )
        self.transform_snapshot = None
        self._applied_affine = None
        self._affine_base_translate = None
        self._update_button_states()

    def get_estimated_affine(self) -> np.ndarray | None:
        """Get the estimated affine in original data coordinates.

        The internal affine maps layer2-world → layer1-world.  This method
        conjugates it so that it operates in each layer's original (unpadded,
        untranslated) data coordinate space:

            A_data = T1_inv @ A_world @ T2

        where T2 translates by +layer2_original_translate and T1_inv by
        -layer1_original_translate.

        Returns:
            Homogeneous affine matrix or None if insufficient valid pairs.
        """
        self._capture_translates()

        affine = self._estimate_affine_transform()
        if affine is None:
            return affine

        ndim = affine.shape[0] - 1

        t1 = self._original_translates.get(self.layer1_name, np.zeros(ndim))[
            :ndim
        ]
        t2 = self._original_translates.get(self.layer2_name, np.zeros(ndim))[
            :ndim
        ]

        # If both translates are zero (no offset), skip conjugation
        if np.allclose(t1, 0) and np.allclose(t2, 0):
            return affine

        # T2 translates by +layer2_translate (data→world)
        T2 = np.eye(ndim + 1)
        T2[:ndim, ndim] = t2

        # T1_inv translates by -layer1_translate (world→data)
        T1_inv = np.eye(ndim + 1)
        T1_inv[:ndim, ndim] = -t1

        return T1_inv @ affine @ T2
