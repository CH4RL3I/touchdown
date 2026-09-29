"""Image degradations, all scaled by a single nuisance level.

Level 0 is a clean image. Level 0.5 is the nominal test condition, level 1.0 is
deliberately harsh (heavy noise, strong lighting changes, large occluders).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Occluder:
    """An elliptical blob lying on the pad plane (a leaf, debris, a passing shadow)."""

    t0: float
    t1: float
    x: float
    y: float
    rx: float
    ry: float
    angle: float

    def polygon(self, n: int = 24) -> np.ndarray:
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ca, sa = np.cos(self.angle), np.sin(self.angle)
        ex, ey = self.rx * np.cos(a), self.ry * np.sin(a)
        return np.stack([self.x + ca * ex - sa * ey, self.y + sa * ex + ca * ey, np.zeros(n)], 1)


@dataclass
class Nuisance:
    level: float
    exposure: float = 0.0  # seconds, drives motion blur from image-plane speed
    defocus_sigma: float = 0.0  # px
    noise_sigma: float = 0.0  # gray levels
    gain_amp: float = 0.0
    gain_freq: float = 0.2
    gain_phase: float = 0.0
    offset_amp: float = 0.0
    grad_strength: float = 0.0
    grad_angle: float = 0.0
    vignette: float = 0.0
    occluders: list[Occluder] = field(default_factory=list)

    @classmethod
    def sample(cls, level: float, rng: np.random.Generator) -> Nuisance:
        L = float(level)
        n = cls(
            level=L,
            exposure=0.010 * L,
            defocus_sigma=0.7 * L,
            noise_sigma=10.0 * L,
            gain_amp=0.35 * L,
            gain_freq=float(rng.uniform(0.1, 0.4)),
            gain_phase=float(rng.uniform(0, 2 * np.pi)),
            offset_amp=30.0 * L,
            grad_strength=0.5 * L,
            grad_angle=float(rng.uniform(0, 2 * np.pi)),
            vignette=0.35 * L,
        )
        n_events = 0 if L <= 0 else (1 if L < 0.75 else 2)
        for _ in range(n_events):
            t0 = float(rng.uniform(1.5, 14.0))
            n.occluders.append(
                Occluder(
                    t0=t0,
                    t1=t0 + float(rng.uniform(1.0, 3.0)),
                    x=float(rng.uniform(-0.45, 0.45)),
                    y=float(rng.uniform(-0.45, 0.45)),
                    rx=float((0.08 + 0.25 * L) * rng.uniform(0.7, 1.3)),
                    ry=float((0.08 + 0.25 * L) * rng.uniform(0.7, 1.3)),
                    angle=float(rng.uniform(0, np.pi)),
                )
            )
        return n

    def active_occluders(self, t: float) -> list[Occluder]:
        return [o for o in self.occluders if o.t0 <= t < o.t1]


def _line_kernel(length: float, angle: float) -> np.ndarray:
    ks = int(np.ceil(length)) | 1
    ks = max(ks, 3)
    k = np.zeros((ks, ks), np.float32)
    c = ks // 2
    dx, dy = np.cos(angle) * length / 2, np.sin(angle) * length / 2
    cv2.line(
        k,
        (int(round(c - dx)), int(round(c - dy))),
        (int(round(c + dx)), int(round(c + dy))),
        1.0,
        1,
    )
    return k / k.sum()


class PhotometricModel:
    """Applies blur, lighting and noise to a clean rendered frame."""

    def __init__(self, width: int, height: int):
        xx, yy = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        self.xx = xx.astype(np.float32)
        self.yy = yy.astype(np.float32)

    def apply(
        self,
        img: np.ndarray,
        nz: Nuisance,
        t: float,
        blur_vec_px: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        out = img.astype(np.float32)
        length = min(float(np.hypot(*blur_vec_px)) * (nz.exposure > 0), 30.0)
        if length >= 1.0:
            out = cv2.filter2D(
                out, -1, _line_kernel(length, float(np.arctan2(blur_vec_px[1], blur_vec_px[0])))
            )
        if nz.defocus_sigma > 0.05:
            out = cv2.GaussianBlur(out, (0, 0), nz.defocus_sigma)
        gain = 1.0 + nz.gain_amp * np.sin(2 * np.pi * nz.gain_freq * t + nz.gain_phase)
        offset = nz.offset_amp * np.sin(2 * np.pi * 0.7 * nz.gain_freq * t + 1.3 * nz.gain_phase)
        if nz.grad_strength > 0 or nz.vignette > 0:
            ramp = nz.grad_strength * (
                self.xx * np.cos(nz.grad_angle) + self.yy * np.sin(nz.grad_angle)
            )
            shade = 1.0 + ramp - nz.vignette * 0.5 * (self.xx**2 + self.yy**2)
            out *= shade
        out = out * gain + offset
        if nz.noise_sigma > 0:
            out += rng.standard_normal(out.shape, dtype=np.float32) * nz.noise_sigma
        return np.clip(out, 0, 255).astype(np.uint8)
