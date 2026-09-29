"""Rotation helpers and frame conventions.

World frame: x east, y north, z up, origin at the centre of the landing pad
(pad plane is z = 0). Camera frame follows OpenCV: x right, y down, z forward.
A nadir-looking camera therefore has camera-to-world rotation ``R_NADIR``.
``R_wc`` always means camera-to-world; ``p`` is the camera position in the world.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

R_NADIR = np.diag([1.0, -1.0, -1.0])
GRAVITY = 9.81


def exp_so3(w: np.ndarray) -> np.ndarray:
    return Rot.from_rotvec(np.asarray(w, dtype=float)).as_matrix()


def log_so3(R: np.ndarray) -> np.ndarray:
    return Rot.from_matrix(R).as_rotvec()


def rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (rad) between two rotation matrices."""
    return float(np.linalg.norm(log_so3(Ra.T @ Rb)))


def attitude_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Camera-to-world rotation of a downward camera tilted by roll/pitch/yaw.

    Angles are applied in the world frame on top of the nadir attitude, so
    (0, 0, 0) is a perfectly level, north-up camera.
    """
    R_pert = Rot.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    return R_pert @ R_NADIR


def tilt_from_nadir(R_wc: np.ndarray) -> float:
    """Angle (rad) between the optical axis and straight down."""
    return float(np.arccos(np.clip(-R_wc[2, 2], -1.0, 1.0)))


def world_to_cam(p: np.ndarray, R_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad-frame points map to the camera as X_c = R_cw X_w + t."""
    R_cw = R_wc.T
    return R_cw, -R_cw @ p
