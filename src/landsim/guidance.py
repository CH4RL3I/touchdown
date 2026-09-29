"""Landing guidance: velocity setpoints from the filtered pad-relative position.

Phases: SEARCH (no estimate yet), ALIGN (lateral error too large to descend),
DESCEND (aligned, speed scheduled with altitude), FINAL (below the commit altitude,
descent continues even if the marker leaves the field of view), and the terminal
states TOUCHDOWN and ABORT. Lateral control is PI on the estimated position;
descent speed is limited by altitude and throttled by lateral error.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .kalman import Estimate


class Phase(StrEnum):
    SEARCH = "search"
    ALIGN = "align"
    DESCEND = "descend"
    FINAL = "final"
    TOUCHDOWN = "touchdown"
    ABORT = "abort"


@dataclass
class GuidanceConfig:
    kp_xy: float = 0.9
    ki_xy: float = 0.12
    i_max: float = 0.5  # m/s cap on the integral contribution
    v_xy_max: float = 2.0  # m/s
    v_desc_max: float = 1.5  # m/s
    v_desc_min: float = 0.35  # m/s
    v_desc_gain: float = 0.3  # 1/s : descent speed = gain * altitude (then clipped)
    align_tol_base: float = 0.15  # m
    align_tol_per_m: float = 0.15  # allowed lateral error grows with altitude
    commit_altitude: float = 0.6  # m: below this the marker may leave the field of view
    touchdown_altitude: float = 0.15  # m
    lost_abort_s: float = 2.0  # marker lost this long above commit altitude -> abort
    search_abort_s: float = 4.0  # never acquired
    near_ground_altitude: float = 1.0  # m
    near_ground_lateral_abort: float = 0.45  # m
    climb_speed: float = 1.0  # m/s on abort


@dataclass
class Command:
    v: np.ndarray  # world-frame velocity setpoint (m/s), z up
    phase: Phase
    reason: str = ""


class LandingGuidance:
    def __init__(self, cfg: GuidanceConfig | None = None):
        self.cfg = cfg or GuidanceConfig()
        self.integral = np.zeros(2)
        self.search_time = 0.0
        self.aborted = False
        self.reason = ""

    def reset(self) -> None:
        self.integral[:] = 0
        self.search_time = 0.0
        self.aborted = False
        self.reason = ""

    def _abort(self, reason: str) -> Command:
        self.aborted, self.reason = True, reason
        return Command(np.array([0.0, 0.0, self.cfg.climb_speed]), Phase.ABORT, reason)

    def step(self, est: Estimate | None, dt: float) -> Command:
        c = self.cfg
        if self.aborted:
            return Command(np.array([0.0, 0.0, c.climb_speed]), Phase.ABORT, self.reason)
        if est is None:
            self.search_time += dt
            if self.search_time > c.search_abort_s:
                return self._abort("marker never acquired")
            return Command(np.zeros(3), Phase.SEARCH)

        z = float(est.p[2])
        lateral = est.p[:2]
        e = float(np.linalg.norm(lateral))

        if z > c.commit_altitude and est.lost_time > c.lost_abort_s:
            return self._abort("marker lost too long")
        # the tolerance widens with staleness: a dead-reckoned estimate is less trustworthy
        stale_scale = 1.0 + min(est.lost_time, 2.0)
        if z < c.near_ground_altitude and e > c.near_ground_lateral_abort * stale_scale:
            return self._abort("excessive lateral error near ground")

        # PI on lateral position with anti-windup by clamping the integral term
        self.integral = np.clip(self.integral - c.ki_xy * lateral * dt, -c.i_max, c.i_max)
        v_xy = -c.kp_xy * lateral + self.integral
        speed = float(np.linalg.norm(v_xy))
        if speed > c.v_xy_max:
            v_xy *= c.v_xy_max / speed

        v_target = float(np.clip(c.v_desc_gain * z, c.v_desc_min, c.v_desc_max))
        if z <= c.commit_altitude:
            phase = Phase.TOUCHDOWN if z <= c.touchdown_altitude else Phase.FINAL
            vz = -c.v_desc_min
        else:
            tol = c.align_tol_base + c.align_tol_per_m * z
            throttle = float(np.clip(2.0 - e / tol, 0.0, 1.0))
            vz = -v_target * throttle
            phase = Phase.DESCEND if throttle >= 1.0 else Phase.ALIGN
        return Command(np.array([v_xy[0], v_xy[1], vz]), phase)
