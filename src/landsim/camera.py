"""Pinhole camera with optional radial/tangential lens distortion."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .geometry import world_to_cam


@dataclass(frozen=True)
class Camera:
    width: int = 640
    height: int = 480
    fx: float = 520.0
    fy: float = 520.0
    cx: float = 321.0
    cy: float = 239.0
    dist: tuple[float, ...] = (-0.06, 0.01, 0.0, 0.0)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def D(self) -> np.ndarray:
        return np.array(self.dist, dtype=float)

    def project(
        self, pts_world: np.ndarray, p: np.ndarray, R_wc: np.ndarray, distort: bool = True
    ) -> np.ndarray:
        """Project world points seen from pose (p, R_wc) to pixel coordinates (N, 2)."""
        R_cw, t = world_to_cam(p, R_wc)
        rvec, _ = cv2.Rodrigues(R_cw)
        D = self.D if distort else np.zeros(4)
        px, _ = cv2.projectPoints(
            np.asarray(pts_world, dtype=float).reshape(-1, 1, 3), rvec, t, self.K, D
        )
        return px.reshape(-1, 2)

    def imperfect_calibration(self) -> Camera:
        """The intrinsics a perception stack would hold after an imperfect calibration."""
        k1, k2, p1, p2 = self.dist
        return replace(
            self,
            fx=self.fx * 1.002,
            fy=self.fy * 1.002,
            cx=self.cx + 0.5,
            cy=self.cy - 0.4,
            dist=(k1 * 1.05, k2, p1, p2),
        )

    def undistort_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """Remap tables that turn a pinhole render into a distorted image."""
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        pts = np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float64)
        und = cv2.undistortPoints(pts, self.K, self.D, P=self.K).reshape(self.height, self.width, 2)
        return und[..., 0].astype(np.float32), und[..., 1].astype(np.float32)
