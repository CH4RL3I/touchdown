"""Projection / PnP round trip and frame-convention checks."""

import cv2
import numpy as np
import pytest

from landsim.camera import Camera
from landsim.geometry import R_NADIR, attitude_from_rpy, rot_angle
from landsim.pad import marker_corners_object


def solve(cam: Camera, pts_obj: np.ndarray, px: np.ndarray):
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        pts_obj, px.astype(np.float64), cam.K, cam.D, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    sols = []
    for rv, tv in zip(rvecs, tvecs, strict=True):
        R_cp, _ = cv2.Rodrigues(rv)
        sols.append((R_cp.T, -R_cp.T @ tv.ravel()))
    return sols


def best(sols, p, R):
    return min(sols, key=lambda s: rot_angle(s[0], R) + np.linalg.norm(s[1] - p))


@pytest.mark.parametrize("side", [1.2, 0.18])
@pytest.mark.parametrize("distort", [False, True])
def test_pnp_round_trip_exact(side, distort):
    cam = Camera() if distort else Camera(dist=(0, 0, 0, 0))
    p = np.array([0.4, -0.3, 3.0])
    R = attitude_from_rpy(np.deg2rad(8), np.deg2rad(-6), np.deg2rad(35))
    obj = marker_corners_object(side)
    px = cam.project(obj, p, R, distort=distort)
    p_hat, R_hat = best(solve(cam, obj, px), p, R)[1], best(solve(cam, obj, px), p, R)[0]
    assert np.linalg.norm(p_hat - p) < 1e-3
    assert np.rad2deg(rot_angle(R_hat, R)) < 0.05


def test_pnp_with_pixel_noise_is_accurate():
    cam = Camera()
    rng = np.random.default_rng(0)
    p = np.array([0.2, 0.1, 2.0])
    R = attitude_from_rpy(np.deg2rad(5), np.deg2rad(4), np.deg2rad(-20))
    obj = marker_corners_object(1.2)
    errs = []
    for _ in range(50):
        px = cam.project(obj, p, R) + rng.normal(0, 0.3, (4, 2))
        errs.append(np.linalg.norm(best(solve(cam, obj, px), p, R)[1] - p))
    assert np.median(errs) < 0.05  # 2.5 percent of range for 0.3 px corner noise


def test_frame_conventions():
    """Nadir camera: pad origin sits on the optical axis, +x is image-right, +y is image-up."""
    cam = Camera(dist=(0, 0, 0, 0))
    p = np.array([0.0, 0.0, 4.0])
    px = cam.project(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), p, R_NADIR, distort=False)
    assert np.allclose(px[0], [cam.cx, cam.cy], atol=1e-6)
    assert px[1][0] > px[0][0] and abs(px[1][1] - px[0][1]) < 1e-6
    assert px[2][1] < px[0][1] and abs(px[2][0] - px[0][0]) < 1e-6
    # ground-plane point appears at the expected scale f * X / Z
    assert px[1][0] - px[0][0] == pytest.approx(cam.fx / 4.0, rel=1e-6)


def test_camera_moving_right_shifts_pad_left():
    cam = Camera(dist=(0, 0, 0, 0))
    a = cam.project(np.zeros((1, 3)), np.array([0.0, 0, 3.0]), R_NADIR, distort=False)
    b = cam.project(np.zeros((1, 3)), np.array([0.5, 0, 3.0]), R_NADIR, distort=False)
    assert b[0, 0] < a[0, 0]
