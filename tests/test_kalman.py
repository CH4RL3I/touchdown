"""Kalman filter consistency (NEES) and dropout behaviour."""

import numpy as np
from scipy.stats import chi2

from landsim.kalman import G_VEC, PoseEstimator, TranslationKF


def test_translation_kf_nees_is_consistent():
    """Filter matched to its own generating model: mean NEES must sit inside the chi^2 band."""
    rng = np.random.default_rng(1)
    n_trials, n_steps, dt = 250, 91, 1 / 30
    sigma_a, sigma_b, sigma_m = 0.3, 0.005, 0.1
    m0 = np.array([1.0, 2.0, 10.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    P0 = np.diag([0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]) ** 2
    L = np.linalg.cholesky(P0)
    nees = np.zeros((n_steps, n_trials))
    R = np.eye(3)
    for j in range(n_trials):
        x_true = m0 + L @ rng.standard_normal(9)
        kf = TranslationKF(sigma_a=sigma_a, sigma_b=sigma_b)
        kf.x, kf.P = m0.copy(), P0.copy()
        for k in range(1, n_steps):
            a_true = np.array([0.5 * np.sin(0.05 * k), 0.3 * np.cos(0.04 * k), -0.4])
            x_true[0:3] += x_true[3:6] * dt + 0.5 * a_true * dt**2
            x_true[3:6] += a_true * dt
            x_true[6:9] += sigma_b * np.sqrt(dt) * rng.standard_normal(3)
            f_b = a_true - G_VEC + x_true[6:9] + rng.normal(0, sigma_a, 3)
            kf.predict(f_b, R, dt)
            if k % 3 == 0:
                z = x_true[:3] + rng.normal(0, sigma_m, 3)
                kf.update(z, np.full(3, sigma_m), gate=None)
            e = x_true - kf.x
            nees[k, j] = e @ np.linalg.solve(kf.P, e)
    lo, hi = chi2.ppf([0.005, 0.995], 9 * n_trials) / n_trials
    for k in (30, 60, 90):
        assert lo < nees[k].mean() < hi, (k, nees[k].mean(), lo, hi)
    # averaged over the whole run the filter is neither over- nor under-confident
    assert 8.0 < nees[10:].mean() < 10.0


def test_covariance_grows_during_dropout_and_shrinks_on_update():
    kf = TranslationKF()
    kf.init(np.array([0.0, 0.0, 5.0]), (0.1, 0.1, 0.1))
    R = np.eye(3)
    f_b = -G_VEC
    stds = []
    for _ in range(60):
        kf.predict(f_b, R, 1 / 30)
        stds.append(kf.pos_std.copy())
    assert all(b[0] >= a[0] for a, b in zip(stds, stds[1:], strict=False))
    before = kf.pos_std.copy()
    kf.update(np.array([0.0, 0.0, 5.0]), np.full(3, 0.05))
    assert np.all(kf.pos_std < before)


def test_gate_rejects_gross_outlier():
    kf = TranslationKF()
    kf.init(np.array([0.0, 0.0, 5.0]), (0.05, 0.05, 0.05))
    x_before = kf.x.copy()
    ok, nis = kf.update(np.array([3.0, -2.0, 9.0]), np.full(3, 0.05))
    assert not ok and nis > 16.27
    assert np.allclose(kf.x, x_before)


def test_estimator_reinitialises_after_repeated_rejections():
    class M:  # minimal VisionMeasurement stand-in
        def __init__(self, p):
            self.p, self.R_wc, self.width_px = np.asarray(p, float), np.eye(3), 100.0

        def position_sigma(self):
            return 0.02, 0.02

    est = PoseEstimator(1 / 30, reinit_after=5)
    f_b = -G_VEC
    est.step(f_b, np.zeros(3), M([0, 0, 5]))
    for _ in range(6):
        est.step(f_b, np.zeros(3), M([4, 4, 8]))
    assert np.allclose(est.estimate().p, [4, 4, 8], atol=0.5)
