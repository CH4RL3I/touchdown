"""One closed-loop landing episode: render -> detect -> filter -> guide -> fly."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .camera import Camera
from .dynamics import Scenario, World
from .geometry import rot_angle
from .guidance import GuidanceConfig, LandingGuidance, Phase
from .kalman import PoseEstimator
from .pad import PadModel
from .perception import MarkerPerception
from .render import Renderer

TOUCHDOWN_SUCCESS_RADIUS = 0.25  # m
MARKER_CODE = {None: 0, "large": 1, "small": 2}


@dataclass
class Assets:
    camera: Camera
    pad: PadModel
    renderer: Renderer


@lru_cache(maxsize=2)
def get_assets(distort: bool = True) -> Assets:
    cam = Camera() if distort else Camera(dist=(0.0, 0.0, 0.0, 0.0))
    pad = PadModel()
    return Assets(cam, pad, Renderer(cam, pad, distort=distort))


@dataclass
class Episode:
    seed: int
    level: float
    outcome: str  # "landed", "abort", "timeout"
    abort_reason: str
    touchdown_error: float  # m, radial, NaN unless landed
    touchdown_xy: np.ndarray
    duration: float
    # per-frame logs, all length N
    t: np.ndarray
    p_true: np.ndarray
    R_true: np.ndarray
    p_raw: np.ndarray
    R_raw: np.ndarray
    marker: np.ndarray  # 0 none, 1 large, 2 small
    p_est: np.ndarray
    R_est: np.ndarray
    est_valid: np.ndarray
    phase: list[str]
    timing_ms: np.ndarray  # columns: perception, filter, guidance
    frames: list[np.ndarray] = field(default_factory=list, repr=False)
    overlays: list[dict] = field(default_factory=list, repr=False)

    @property
    def success(self) -> bool:
        return self.outcome == "landed" and self.touchdown_error <= TOUCHDOWN_SUCCESS_RADIUS

    def lean(self) -> Episode:
        """Copy without image data, for cheap transfer between processes."""
        from dataclasses import replace

        return replace(self, frames=[], overlays=[])


def run_episode(
    seed: int,
    level: float = 0.5,
    guidance_cfg: GuidanceConfig | None = None,
    record_frames: bool = False,
    max_time: float = 60.0,
    distort: bool = True,
) -> Episode:
    assets = get_assets(distort)
    cam, pad, renderer = assets.camera, assets.pad, assets.renderer
    scn = Scenario.sample(seed, level)
    world = World(scn)
    dt = scn.dt
    perception = MarkerPerception(cam.imperfect_calibration(), pad)
    estimator = PoseEstimator(dt)
    guidance = LandingGuidance(guidance_cfg)
    rng_img = scn.rng_streams["img"]

    logs: dict[str, list] = {
        k: []
        for k in ["t", "p", "R", "p_raw", "R_raw", "mk", "p_est", "R_est", "valid", "phase", "tm"]
    }
    frames: list[np.ndarray] = []
    overlays: list[dict] = []
    nan3, nan33 = np.full(3, np.nan), np.full((3, 3), np.nan)

    imu = None
    outcome, reason = "timeout", ""
    n_steps = int(max_time / dt)
    for _ in range(n_steps):
        v_true = world.v
        gray = renderer.render(
            world.p, world.R, scn.nuisance, world.t, v_true, world.omega_b, rng_img
        )

        t0 = time.process_time()
        if imu is not None:
            estimator.predict(imu.f_b, imu.w_b)
        est_prior = estimator.estimate()
        t1 = time.process_time()
        prior = None if est_prior is None else (est_prior.p, est_prior.R_wc)
        meas = perception.measure(gray, prior)
        t2 = time.process_time()
        estimator.update(meas)
        est = estimator.estimate()
        t3 = time.process_time()
        cmd = guidance.step(est, dt)
        t4 = time.process_time()

        logs["t"].append(world.t)
        logs["p"].append(world.p.copy())
        logs["R"].append(world.R.copy())
        logs["p_raw"].append(meas.p if meas else nan3)
        logs["R_raw"].append(meas.R_wc if meas else nan33)
        logs["mk"].append(MARKER_CODE[meas.marker if meas else None])
        logs["p_est"].append(est.p if est else nan3)
        logs["R_est"].append(est.R_wc if est else nan33)
        logs["valid"].append(est is not None)
        logs["phase"].append(cmd.phase.value)
        logs["tm"].append([(t2 - t1 + 0.0) * 1e3, ((t1 - t0) + (t3 - t2)) * 1e3, (t4 - t3) * 1e3])
        if record_frames:
            frames.append(gray)
            overlays.append(
                {
                    "corners": None if meas is None else meas.corners,
                    "marker": None if meas is None else meas.marker,
                    "est_p": None if est is None else est.p,
                    "est_R": None if est is None else est.R_wc,
                    "raw_p": None if meas is None else meas.p,
                    "raw_R": None if meas is None else meas.R_wc,
                    "phase": cmd.phase.value,
                    "roi": perception.last_roi,
                    "t": world.t,
                    "p_true": world.p.copy(),
                }
            )

        if cmd.phase is Phase.ABORT:
            outcome, reason = "abort", cmd.reason
            break
        imu = world.step(cmd.v)
        if world.touched_down:
            outcome = "landed"
            break

    td_xy = np.full(2, np.nan)
    td_err = float("nan")
    if outcome == "landed" and world.touchdown_p is not None:
        td_xy = world.touchdown_p[:2].copy()
        td_err = float(np.linalg.norm(td_xy))
    return Episode(
        seed=seed,
        level=level,
        outcome=outcome,
        abort_reason=reason,
        touchdown_error=td_err,
        touchdown_xy=td_xy,
        duration=world.t,
        t=np.array(logs["t"]),
        p_true=np.array(logs["p"]),
        R_true=np.array(logs["R"]),
        p_raw=np.array(logs["p_raw"]),
        R_raw=np.array(logs["R_raw"]),
        marker=np.array(logs["mk"]),
        p_est=np.array(logs["p_est"]),
        R_est=np.array(logs["R_est"]),
        est_valid=np.array(logs["valid"]),
        phase=logs["phase"],
        timing_ms=np.array(logs["tm"]),
        frames=frames,
        overlays=overlays,
    )


def pose_errors(ep: Episode) -> dict[str, np.ndarray]:
    """Per-frame position (m) and attitude (deg) errors for raw PnP and the filter."""
    out = {}
    for name, p, R in [("raw", ep.p_raw, ep.R_raw), ("filt", ep.p_est, ep.R_est)]:
        pos = np.linalg.norm(p - ep.p_true, axis=1)
        att = np.full(len(pos), np.nan)
        for i in np.flatnonzero(~np.isnan(pos)):
            att[i] = np.rad2deg(rot_angle(R[i], ep.R_true[i]))
        out[f"{name}_pos"], out[f"{name}_att"] = pos, att
    return out


__all__ = ["Episode", "run_episode", "pose_errors", "get_assets"]
