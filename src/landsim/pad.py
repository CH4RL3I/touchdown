"""The landing pad: a nested ArUco board plus a textured ground plane.

A large marker (visible from altitude) has a white 2x2-cell hole in its middle
in which a small marker (visible close up) sits. Both share the pad centre, so
either one yields the pose of the same pad frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DICT_NAME = cv2.aruco.DICT_4X4_250
BLACK, WHITE = 20, 235


def marker_corners_object(side: float) -> np.ndarray:
    """Marker corners in the pad frame, ordered TL, TR, BR, BL as seen from above.

    This is the ordering ``SOLVEPNP_IPPE_SQUARE`` expects and the one ArUco returns.
    """
    h = side / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def _central_cells_white(dictionary, marker_id: int) -> bool:
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, 60, borderBits=1)
    cells = img.reshape(6, 10, 6, 10).mean(axis=(1, 3))
    return bool((cells[2:4, 2:4] > 127).all())


def _ground_texture(size_px: int, seed: int = 7) -> np.ndarray:
    """Multi-scale noise resembling gravel/grass, with a few dark and bright distractors."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size_px, size_px), np.float32)
    for sigma, amp in [(64, 22), (16, 16), (4, 10), (1.2, 6)]:
        n = rng.normal(size=(size_px, size_px)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        img += amp * n / (n.std() + 1e-6)
    img += 95
    ppm = size_px / 8.0
    for _ in range(9):  # dark/bright rectangles that are not the pad
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(1.3, 3.6)
        cx, cy = size_px / 2 + ppm * r * np.cos(ang), size_px / 2 + ppm * r * np.sin(ang)
        w, h = rng.uniform(0.15, 0.7, 2) * ppm
        box = cv2.boxPoints(((cx, cy), (w, h), rng.uniform(0, 90))).astype(np.int32)
        cv2.fillConvexPoly(img, box, float(rng.choice([35, 45, 170, 190])))
    return np.clip(img, 0, 255).astype(np.uint8)


def _pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    out = [img]
    for _ in range(levels):
        out.append(cv2.pyrDown(out[-1]))
    return out


@dataclass
class PadModel:
    pad_size: float = 1.8
    large_side: float = 1.2
    small_side: float = 0.18
    ppm: float = 1200.0  # texture pixels per metre at pyramid level 0
    ground_size: float = 8.0
    ground_px: int = 2048
    large_pick: int = 0  # which of the qualifying dictionary ids to use as the large marker
    small_pick: int = 1
    large_id: int = field(init=False)
    small_id: int = field(init=False)
    pad_pyr: list[np.ndarray] = field(init=False, repr=False)
    ground_pyr: list[np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(DICT_NAME)
        candidates = [i for i in range(250) if _central_cells_white(dictionary, i)]
        self.large_id, self.small_id = candidates[self.large_pick], candidates[self.small_pick]
        n = int(round(self.pad_size * self.ppm))
        big = int(round(self.large_side * self.ppm))
        small = int(round(self.small_side * self.ppm))
        assert n % 2 == 0 and big % 12 == 0 and small % 12 == 0
        tex = np.full((n, n), WHITE, np.uint8)
        for side_px, mid in [(big, self.large_id), (small, self.small_id)]:
            m = cv2.aruco.generateImageMarker(dictionary, mid, side_px, borderBits=1)
            m = np.where(m > 127, WHITE, BLACK).astype(np.uint8)
            a = (n - side_px) // 2
            tex[a : a + side_px, a : a + side_px] = m
        alpha = np.full_like(tex, 255)
        levels = 7
        self.pad_pyr = _pyramid(np.dstack([tex, alpha]), levels)
        self.ground_pyr = _pyramid(_ground_texture(self.ground_px), levels)
        self.dictionary = dictionary

    @property
    def ground_ppm(self) -> float:
        return self.ground_px / self.ground_size
