"""Simulator determinism and end-to-end perception sanity."""

import numpy as np

from landsim.dynamics import Scenario, World
from landsim.geometry import attitude_from_rpy, rot_angle
from landsim.perception import MarkerPerception


def rollout(assets, seed, steps=12):
    scn = Scenario.sample(seed, 0.5)
    w = World(scn)
    rng = scn.rng_streams["img"]
    frames, states = [], []
    for _ in range(steps):
        frames.append(assets.renderer.render(w.p, w.R, scn.nuisance, w.t, w.v, w.omega_b, rng))
        imu = w.step(np.array([0.1, -0.1, -1.0]))
        states.append(np.r_[w.p, w.v, imu.f_b, imu.w_b])
    return np.array(frames), np.array(states)


def test_same_seed_is_bit_identical(assets):
    f1, s1 = rollout(assets, 11)
    f2, s2 = rollout(assets, 11)
    assert np.array_equal(f1, f2)
    assert np.array_equal(s1, s2)


def test_different_seeds_differ(assets):
    f1, s1 = rollout(assets, 11)
    f2, s2 = rollout(assets, 12)
    assert not np.array_equal(s1, s2)
    assert not np.array_equal(f1, f2)


def test_nuisance_level_zero_is_clean_and_deterministic(assets):
    scn = Scenario.sample(3, 0.0)
    assert scn.nuisance.noise_sigma == 0 and not scn.nuisance.occluders


def test_touchdown_is_detected_and_interpolated():
    scn = Scenario.sample(4, 0.0)
    scn.p0 = np.array([0.3, -0.2, 0.5])
    w = World(scn)
    for _ in range(200):
        w.step(np.array([0.0, 0.0, -1.0]))
        if w.touched_down:
            break
    assert w.touched_down
    assert abs(w.touchdown_p[2] - World.z_touchdown) < 1e-9


def test_end_to_end_detection_recovers_pose(assets):
    """Render a clean frame at a known pose, detect the markers, recover the pose."""
    perception = MarkerPerception(assets.camera, assets.pad)
    for h, marker in [(4.0, "large"), (0.9, "small")]:
        p = np.array([0.15, -0.1, h])
        R = attitude_from_rpy(np.deg2rad(4), np.deg2rad(-3), np.deg2rad(25))
        img = assets.renderer.render_clean(p, R)
        m = perception.measure(img, (p, R))
        assert m is not None and m.marker == marker
        assert np.linalg.norm(m.p - p) < 0.03 * h
        assert np.rad2deg(rot_angle(m.R_wc, R)) < 3.0
