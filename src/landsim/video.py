"""Annotated descent video/GIF: detected marker, filtered pose axes, HUD, top-down inset."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .closed_loop import Episode, get_assets, run_episode

GREEN, CYAN, YELLOW, RED, WHITE = (
    (60, 220, 60),
    (255, 220, 0),
    (0, 220, 255),
    (60, 60, 255),
    (255, 255, 255),
)


def _axes(img, cam, R_wc, p, length):
    R_cp = R_wc.T
    rvec, _ = cv2.Rodrigues(R_cp)
    cv2.drawFrameAxes(img, cam.K, cam.D, rvec, -R_cp @ p, length, 3)


def annotate(ep: Episode, i: int, cam, trail: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    ov = ep.overlays[i]
    img = cv2.cvtColor(ep.frames[i], cv2.COLOR_GRAY2BGR)
    if ov["roi"] is not None:
        x0, y0, x1, y1 = ov["roi"]
        cv2.rectangle(img, (x0, y0), (x1, y1), (150, 150, 150), 1)
    if ov["corners"] is not None:
        cv2.polylines(img, [ov["corners"].astype(np.int32).reshape(-1, 1, 2)], True, GREEN, 2)
    est_p, est_R = ov["est_p"], ov["est_R"]
    if est_p is not None:
        length = float(np.clip(0.25 * est_p[2] / 3.0 + 0.1, 0.12, 0.6))
        _axes(img, cam, est_R, est_p, length)
    true_px = cam.project(np.zeros((1, 3)), ov["p_true"], ep.R_true[i]).ravel()
    cv2.drawMarker(img, tuple(int(v) for v in true_px), RED, cv2.MARKER_CROSS, 14, 1)

    # HUD
    z_true = float(ov["p_true"][2])
    lat_true = float(np.linalg.norm(ov["p_true"][:2]))
    lines = [
        f"phase  {ov['phase']}",
        f"alt    {z_true:5.2f} m" + (f"  (est {est_p[2]:.2f})" if est_p is not None else ""),
        f"offset {lat_true:5.2f} m"
        + (f"  (est {np.linalg.norm(est_p[:2]):.2f})" if est_p is not None else ""),
        f"vision {ov['marker'] or 'lost'}",
    ]
    box = img.copy()
    cv2.rectangle(box, (6, 6), (250, 12 + 20 * len(lines)), (0, 0, 0), -1)
    img = cv2.addWeighted(box, 0.55, img, 0.45, 0)
    for k, s in enumerate(lines):
        col = YELLOW if (k == 3 and ov["marker"] is None) else WHITE
        cv2.putText(img, s, (12, 26 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # top-down inset (true trail white, estimate cyan), fixed +-3.5 m
    S, m = 130, 3.5
    ins = np.full((S, S, 3), 30, np.uint8)
    to = lambda xy: (int(S / 2 + xy[0] / m * S / 2), int(S / 2 - xy[1] / m * S / 2))  # noqa: E731
    cv2.rectangle(ins, to((-0.9, 0.9)), to((0.9, -0.9)), (200, 200, 200), 1)
    cv2.circle(ins, to((0, 0)), max(int(0.25 / m * S / 2), 2), RED, 1)
    for a, b in zip(trail[:i], trail[1 : i + 1], strict=False):
        cv2.line(ins, to(a[:2]), to(b[:2]), (170, 170, 170), 1)
    cv2.circle(ins, to(ov["p_true"][:2]), 3, WHITE, -1)
    if est_p is not None:
        cv2.circle(ins, to(est_p[:2]), 5, CYAN, 1)
    cv2.putText(ins, "top view", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)
    h, w = img.shape[:2]
    img[h - S - 6 : h - 6, w - S - 6 : w - 6] = ins
    cv2.putText(
        img,
        f"t = {ov['t']:4.1f} s",
        (w - S - 6, h - S - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def write_video(
    seed: int,
    level: float,
    gif: Path | None,
    mp4: Path | None,
    stride: int = 4,
    size: tuple[int, int] = (360, 270),
    colors: int = 32,
    max_time: float = 60.0,
) -> Episode:
    ep = run_episode(seed, level, record_frames=True, max_time=max_time)
    cam = get_assets(True).camera
    trail = ep.p_true
    idx = list(range(0, len(ep.frames), stride))
    if idx[-1] != len(ep.frames) - 1:
        idx.append(len(ep.frames) - 1)
    frames = [annotate(ep, i, cam, trail, size) for i in idx]
    if mp4 is not None:
        mp4.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30.0 / stride, size)
        for f in frames:
            vw.write(f)
        vw.release()
    if gif is not None:
        gif.parent.mkdir(parents=True, exist_ok=True)
        pil = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(
                colors, dither=Image.Dither.NONE
            )
            for f in frames
        ]
        dur = int(round(1000 * stride / 30.0))
        durations = [dur] * (len(pil) - 1) + [1500]  # hold the last frame
        pil[0].save(
            gif, save_all=True, append_images=pil[1:], duration=durations, loop=0, optimize=True
        )
    return ep
