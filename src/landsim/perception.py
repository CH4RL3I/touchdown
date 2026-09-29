"""Marker detection and pose recovery (ArUco + IPPE PnP)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Camera
from .geometry import rot_angle
from .pad import PadModel, marker_corners_object


@dataclass
class VisionMeasurement:
    marker: str  # "large" or "small"
    p: np.ndarray  # camera position in the pad frame (m)
    R_wc: np.ndarray  # camera-to-pad rotation
    corners: np.ndarray  # (4, 2) detected pixel corners
    width_px: float  # mean edge length of the detected marker
    reproj_px: float  # RMS reprojection error of the chosen PnP solution

    def position_sigma(
        self, h_est: float | None = None, sigma_corner: float = 1.0, f: float = 520.0
    ):
        """Rough 1-sigma position error (m) as (lateral, vertical).

        Corner noise maps to lateral error via h/f and to attitude-coupled error
        via h/width; depth from a small planar target is the weakest axis.
        """
        h = max(float(self.p[2]) if h_est is None else h_est, 0.1)
        w = max(self.width_px, 4.0)
        lateral = h * sigma_corner * (1.0 / f + 1.5 / w)
        vertical = h * sigma_corner * 1.5 / w
        return max(lateral, 0.005), max(vertical, 0.005)


class MarkerPerception:
    """Detects the nested markers and returns a pad-relative camera pose.

    Marker choice switches by altitude with hysteresis (large above ``switch_hi``,
    small below ``switch_lo``); if the preferred marker is not detected the other
    one is used when available. When a pose prior is supplied (from the filter) and
    the marker was seen recently, detection runs on a region of interest around the
    predicted pad footprint, which is what makes the far-range frames cheap.
    """

    def __init__(
        self,
        camera: Camera,
        pad: PadModel,
        switch_lo: float = 1.6,
        switch_hi: float = 2.0,
        half_res_below: float = 3.0,
        roi_max_misses: int = 2,
        ambiguity_ratio: float = 1.3,
    ):
        self.cam = camera
        self.pad = pad
        self.switch_lo, self.switch_hi = switch_lo, switch_hi
        self.half_res_below = half_res_below
        self.roi_max_misses = roi_max_misses
        self.ambiguity_ratio = ambiguity_ratio
        self.prefer = "large"
        self.misses = 99
        self.last_roi: tuple[int, int, int, int] | None = None

        # The small marker sits inside the large one, which confuses the stock candidate
        # filter at mid range (the pad-edge quad swallows the marker quad). ArUco3 detection
        # works on a canonical-size image and does not suffer from this, so the large marker
        # gets its own ArUco3 detector; the small one uses the standard detector. Corners are
        # refined by our own cornerSubPix pass because ArUco3 mode skips refinement.
        def make(aruco3: bool):
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
            params.adaptiveThreshWinSizeMin = 23
            params.adaptiveThreshWinSizeMax = 23
            params.minMarkerPerimeterRate = 0.02
            params.errorCorrectionRate = 0.5
            params.useAruco3Detection = aruco3
            if aruco3:
                params.minMarkerLengthRatioOriginalImg = 0.03
            return cv2.aruco.ArucoDetector(pad.dictionary, params)

        self.det_large = make(True)
        self.det_small = make(False)
        self.obj = {
            "large": marker_corners_object(pad.large_side),
            "small": marker_corners_object(pad.small_side),
        }

    def _roi(self, prior: tuple[np.ndarray, np.ndarray]) -> tuple[int, int, int, int] | None:
        p, R_wc = prior
        if p[2] <= 0.2:
            return None
        h = self.pad.pad_size / 2
        foot = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
        px = self.cam.project(foot, p, R_wc)
        lo, hi = px.min(0), px.max(0)
        margin = 0.35 * (hi - lo).max() + 24
        x0, y0 = np.maximum(lo - margin, 0).astype(int)
        x1 = int(min(hi[0] + margin, self.cam.width))
        y1 = int(min(hi[1] + margin, self.cam.height))
        if (
            x1 - x0 < 64
            or y1 - y0 < 64
            or (x1 - x0) * (y1 - y0) > 0.6 * self.cam.width * self.cam.height
        ):
            return None
        return int(x0), int(y0), x1, y1

    def _detect(
        self, gray: np.ndarray, half: bool, altitude_hint: float | None, offset: np.ndarray
    ) -> dict[str, np.ndarray]:
        if half:
            src = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            scale = 2.0
        else:
            src, scale = gray, 1.0
        runs = []
        if altitude_hint is None or altitude_hint > 1.2:
            runs.append(self.det_large)
        if altitude_hint is None or altitude_hint < 7.0:
            runs.append(self.det_small)
        names = {self.pad.large_id: "large", self.pad.small_id: "small"}
        found: dict[str, np.ndarray] = {}
        for det in runs:
            corners, ids, _ = det.detectMarkers(src)
            if ids is None:
                continue
            for c, i in zip(corners, ids.ravel(), strict=True):
                name = names.get(int(i))
                if name is not None and name not in found:
                    c = c.reshape(4, 2).astype(np.float32) * scale
                    found[name] = self._refine(gray, c) + offset.astype(np.float32)
        return found

    def _pose(
        self, name: str, corners: np.ndarray, R_prior: np.ndarray | None
    ) -> VisionMeasurement | None:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            self.obj[name], corners, self.cam.K, self.cam.D, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if n == 0:
            return None
        cands = []
        for rv, tv, e in zip(rvecs, tvecs, errs, strict=True):
            R_cp, _ = cv2.Rodrigues(rv)
            R_wc = R_cp.T
            p = -R_cp.T @ tv.ravel()
            if p[2] <= 0.0:  # camera below the pad plane is unphysical
                continue
            cands.append((float(np.ravel(e)[0]), R_wc, p))
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        best = cands[0]
        # IPPE returns two candidates. Trust the lower reprojection error unless the two
        # are nearly tied, in which case the attitude prior breaks the tie.
        if (
            R_prior is not None
            and len(cands) > 1
            and cands[1][0] < self.ambiguity_ratio * cands[0][0]
        ):
            best = min(cands, key=lambda c: rot_angle(c[1], R_prior))
        edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        return VisionMeasurement(name, best[2], best[1], corners, float(edges.mean()), best[0])

    def measure(
        self, gray: np.ndarray, prior: tuple[np.ndarray, np.ndarray] | None = None
    ) -> VisionMeasurement | None:
        """Detect and localise. ``prior`` is the filter's (position, camera-to-world) estimate."""
        hint = None if prior is None else float(prior[0][2])
        R_prior = None if prior is None else prior[1]
        if hint is not None:
            if hint < self.switch_lo:
                self.prefer = "small"
            elif hint > self.switch_hi:
                self.prefer = "large"
        roi = self._roi(prior) if prior is not None and self.misses <= self.roi_max_misses else None
        self.last_roi = roi
        if roi is not None:
            x0, y0, x1, y1 = roi
            img = np.ascontiguousarray(gray[y0:y1, x0:x1])
            offset = np.array([x0, y0])
        else:
            img, offset = gray, np.zeros(2)
        half = hint is not None and hint < self.half_res_below
        found = self._detect(img, half, hint, offset)
        result = None
        for name in (self.prefer, "small" if self.prefer == "large" else "large"):
            if name in found:
                result = self._pose(name, found[name], R_prior)
                if result is not None:
                    break
        self.misses = 0 if result is not None else self.misses + 1
        return result

    @staticmethod
    def _refine(gray: np.ndarray, c: np.ndarray) -> np.ndarray:
        edge = float(np.linalg.norm(c[1] - c[0]))
        win = int(np.clip(0.4 * edge / 6.0, 2, 10))
        c = c.reshape(-1, 1, 2).astype(np.float32).copy()
        cv2.cornerSubPix(
            gray,
            c,
            (win, win),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        return c.reshape(4, 2)
