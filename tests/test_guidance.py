"""Guidance saturation, phase logic and abort conditions."""

import numpy as np

from landsim.guidance import GuidanceConfig, LandingGuidance, Phase
from landsim.kalman import Estimate

DT = 1 / 30


def est(x=0.0, y=0.0, z=5.0, lost=0.0):
    return Estimate(
        p=np.array([x, y, z]),
        v=np.zeros(3),
        R_wc=np.eye(3),
        pos_std=np.full(3, 0.05),
        age=int(lost / DT),
        lost_time=lost,
    )


def test_lateral_command_saturates_and_points_at_pad():
    g = LandingGuidance()
    c = g.step(est(x=40.0, y=-30.0), DT)
    xy = c.v[:2]
    assert np.linalg.norm(xy) <= g.cfg.v_xy_max + 1e-9
    assert xy[0] < 0 < xy[1]  # towards the pad
    assert np.isclose(np.linalg.norm(xy), g.cfg.v_xy_max)


def test_descent_speed_is_bounded_and_schedules_with_altitude():
    cfg = GuidanceConfig()
    g = LandingGuidance(cfg)
    high = g.step(est(z=50.0), DT).v[2]
    low = g.step(est(z=1.5), DT).v[2]
    assert -cfg.v_desc_max - 1e-9 <= high < 0
    assert high < low < 0  # descends faster when high
    assert abs(low) >= cfg.v_desc_min


def test_no_descent_while_misaligned():
    g = LandingGuidance()
    c = g.step(est(x=2.0, y=0.0, z=2.0), DT)
    assert c.phase is Phase.ALIGN and abs(c.v[2]) < 1e-9


def test_aligned_descends():
    g = LandingGuidance()
    c = g.step(est(x=0.02, y=0.0, z=4.0), DT)
    assert c.phase is Phase.DESCEND and c.v[2] < 0


def test_abort_when_marker_lost_too_long():
    g = LandingGuidance()
    assert g.step(est(z=4.0, lost=1.0), DT).phase is not Phase.ABORT
    c = g.step(est(z=4.0, lost=g.cfg.lost_abort_s + 0.1), DT)
    assert c.phase is Phase.ABORT and c.v[2] > 0
    # abort latches even if the marker comes back
    assert g.step(est(z=4.0), DT).phase is Phase.ABORT


def test_no_lost_abort_below_commit_altitude():
    g = LandingGuidance()
    c = g.step(est(z=0.4, lost=5.0), DT)
    assert c.phase in (Phase.FINAL, Phase.TOUCHDOWN) and c.v[2] < 0


def test_abort_on_excessive_lateral_error_near_ground():
    g = LandingGuidance()
    c = g.step(est(x=0.8, z=0.8), DT)
    assert c.phase is Phase.ABORT and "lateral" in c.reason
    # the same offset high up is merely corrected
    assert LandingGuidance().step(est(x=0.8, z=6.0), DT).phase is not Phase.ABORT


def test_search_then_abort_if_never_acquired():
    g = LandingGuidance()
    assert g.step(None, DT).phase is Phase.SEARCH
    for _ in range(int(g.cfg.search_abort_s / DT) + 2):
        c = g.step(None, DT)
    assert c.phase is Phase.ABORT


def test_integral_is_clamped():
    g = LandingGuidance()
    for _ in range(3000):
        g.step(est(x=5.0, z=5.0), DT)
    assert np.all(np.abs(g.integral) <= g.cfg.i_max + 1e-9)
