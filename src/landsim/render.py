"""Frame synthesis: pad + ground rendered through a pinhole camera."""

from __future__ import annotations

import cv2
import numpy as np

from .camera import Camera
from .geometry import world_to_cam
from .nuisance import Nuisance, PhotometricModel
from .pad import PadModel


class Renderer:
    def __init__(self, camera: Camera, pad: PadModel, distort: bool = True):
        self.cam = camera
        self.pad = pad
        self.distort = distort
        self.photo = PhotometricModel(camera.width, camera.height)
        self.map_x, self.map_y = camera.undistort_maps() if distort else (None, None)
        n_pad = pad.pad_pyr[0].shape[0]
        n_gnd = pad.ground_pyr[0].shape[0]
        self._c_pad, self._c_gnd = (n_pad - 1) / 2.0, (n_gnd - 1) / 2.0

    @staticmethod
    def _tex_to_world(scale: float, centre: float, ppm: float) -> np.ndarray:
        # texture pixel (u, v) at pyramid scale s (level-0 coordinate = s * u)
        return np.array(
            [[scale / ppm, 0, -centre / ppm], [0, -scale / ppm, centre / ppm], [0, 0, 1.0]]
        )

    def render_clean(
        self, p: np.ndarray, R_wc: np.ndarray, nuisance: Nuisance | None = None, t: float = 0.0
    ) -> np.ndarray:
        cam = self.cam
        R_cw, tvec = world_to_cam(p, R_wc)
        H_plane = cam.K @ np.column_stack([R_cw[:, 0], R_cw[:, 1], tvec])
        depth = max(float(tvec[2]), 0.05)
        img_scale = cam.fx / depth  # image px per metre near the pad centre
        size = (cam.width, cam.height)

        def level(ppm: float, n_levels: int) -> int:
            return int(np.clip(np.floor(np.log2(ppm / img_scale)), 0, n_levels - 1))

        lg = level(self.pad.ground_ppm, len(self.pad.ground_pyr))
        A = self._tex_to_world(2.0**lg, self._c_gnd, self.pad.ground_ppm)
        ground = cv2.warpPerspective(
            self.pad.ground_pyr[lg],
            H_plane @ A,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float32)

        lp = level(self.pad.ppm, len(self.pad.pad_pyr))
        A = self._tex_to_world(2.0**lp, self._c_pad, self.pad.ppm)
        padw = cv2.warpPerspective(
            self.pad.pad_pyr[lp],
            H_plane @ A,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32)
        a = padw[..., 1:2] / 255.0
        img = ground * (1 - a[..., 0]) + padw[..., 0] * a[..., 0]

        if nuisance is not None:
            for occ in nuisance.active_occluders(t):
                poly = self.cam.project(occ.polygon(), p, R_wc, distort=False)
                mask = np.zeros(img.shape, np.float32)
                cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1.0)
                mask = cv2.GaussianBlur(mask, (0, 0), 1.5) * 0.93
                img = img * (1 - mask) + 40.0 * mask
        img = np.clip(img, 0, 255).astype(np.uint8)
        if self.distort:
            img = cv2.remap(
                img, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
        return img

    def blur_vector(
        self, p: np.ndarray, R_wc: np.ndarray, v: np.ndarray, omega_b: np.ndarray, exposure: float
    ) -> np.ndarray:
        """Image-plane displacement (px) of the pad centre during one exposure."""
        from .geometry import exp_so3

        origin = np.zeros((1, 3))
        a = self.cam.project(origin, p, R_wc)
        R2 = R_wc @ exp_so3(omega_b * exposure)
        b = self.cam.project(origin, p + v * exposure, R2)
        d = (b - a).ravel()
        return d / max(exposure, 1e-9) * exposure  # displacement over the exposure

    def render(
        self,
        p: np.ndarray,
        R_wc: np.ndarray,
        nuisance: Nuisance,
        t: float,
        v: np.ndarray,
        omega_b: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        clean = self.render_clean(p, R_wc, nuisance, t)
        blur = self.blur_vector(p, R_wc, v, omega_b, max(nuisance.exposure, 1e-6))
        return self.photo.apply(clean, nuisance, t, blur, rng)
