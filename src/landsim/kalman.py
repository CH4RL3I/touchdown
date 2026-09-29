"""State estimation: a linear Kalman filter for position/velocity/accel-bias plus a
complementary attitude filter, fusing gyro/accelerometer with vision poses.

Translation state x = [p (3), v (3), b (3)] with p, v in the pad frame (m, m/s) and b the
accelerometer bias in the body frame. The IMU drives the prediction:
    a = R (f_b - b) + g,   p+ = p + v dt + a dt^2 / 2,   v+ = v + a dt
where R is the attitude estimate. Because R changes every step the transition matrix is
time-varying, but the filter stays linear in x, so no linearisation error appears.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import GRAVITY, exp_so3, log_so3
from .perception import VisionMeasurement

G_VEC = np.array([0.0, 0.0, -GRAVITY])
CHI2_3DOF_999 = 16.27


class TranslationKF:
    def __init__(self, sigma_a: float = 0.3, sigma_b: float = 0.005):
        self.sigma_a = sigma_a  # accel noise incl. attitude-error leakage (m/s^2)
        self.sigma_b = sigma_b  # accel bias random walk (m/s^2 / sqrt(s))
        self.x = np.zeros(9)
        self.P = np.eye(9)

    def init(self, p, pos_sigma, sigma_v0=1.0, sigma_b0=0.15) -> None:
        self.x = np.zeros(9)
        self.x[:3] = p
        self.P = np.diag(np.r_[np.asarray(pos_sigma) ** 2, [sigma_v0**2] * 3, [sigma_b0**2] * 3])

    def predict(self, f_b: np.ndarray, R_wb: np.ndarray, dt: float) -> None:
        x = self.x
        a = R_wb @ (f_b - x[6:9]) + G_VEC
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * dt**2 * R_wb
        F[3:6, 6:9] = -dt * R_wb
        x[0:3] += x[3:6] * dt + 0.5 * a * dt**2
        x[3:6] += a * dt
        G = np.zeros((9, 3))
        G[0:3] = 0.5 * dt**2 * np.eye(3)
        G[3:6] = dt * np.eye(3)
        Q = self.sigma_a**2 * G @ G.T
        Q[6:9, 6:9] += self.sigma_b**2 * dt * np.eye(3)
        self.P = F @ self.P @ F.T + Q

    def update(
        self, z: np.ndarray, sigma: np.ndarray, gate: float | None = CHI2_3DOF_999
    ) -> tuple[bool, float]:
        """Position update; returns (accepted, NIS)."""
        H = np.zeros((3, 9))
        H[:, :3] = np.eye(3)
        Rm = np.diag(np.asarray(sigma, dtype=float) ** 2)
        y = z - self.x[:3]
        S = H @ self.P @ H.T + Rm
        nis = float(y @ np.linalg.solve(S, y))
        if gate is not None and nis > gate:
            return False, nis
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T  # Joseph form
        return True, nis

    @property
    def pos_std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.P)[:3])


class AttitudeFilter:
    """Gyro integration corrected by vision attitude (complementary/Mahony style)."""

    def __init__(self, bias_gain: float = 0.02, max_innovation_deg: float = 25.0):
        self.R = np.eye(3)
        self.bias = np.zeros(3)
        self.bias_gain = bias_gain
        self.max_innov = np.deg2rad(max_innovation_deg)

    def init(self, R: np.ndarray) -> None:
        self.R = R.copy()
        self.bias = np.zeros(3)

    def predict(self, w_meas: np.ndarray, dt: float) -> None:
        self.R = self.R @ exp_so3((w_meas - self.bias) * dt)

    def update(self, R_meas: np.ndarray, gain: float, dt: float) -> bool:
        delta = log_so3(self.R.T @ R_meas)
        if np.linalg.norm(delta) > self.max_innov:
            return False
        self.R = self.R @ exp_so3(gain * delta)
        self.bias = np.clip(self.bias - self.bias_gain * gain * delta / dt, -0.02, 0.02)
        return True


@dataclass
class Estimate:
    p: np.ndarray
    v: np.ndarray
    R_wc: np.ndarray
    pos_std: np.ndarray
    age: int  # frames since the last accepted vision update
    lost_time: float  # seconds since the last accepted vision update


class PoseEstimator:
    """Vision + IMU fusion with dropout handling and outlier gating."""

    def __init__(self, dt: float, sigma_inflation: float = 1.5, reinit_after: int = 10):
        self.dt = dt
        self.kf = TranslationKF()
        self.att = AttitudeFilter()
        self.sigma_inflation = sigma_inflation
        self.reinit_after = reinit_after
        self.initialised = False
        self.age = 0
        self.rejects = 0

    def predict(self, f_b: np.ndarray, w_b: np.ndarray) -> None:
        if self.initialised:
            self.att.predict(w_b, self.dt)
            self.kf.predict(f_b, self.att.R, self.dt)
            self.age += 1

    def update(self, meas: VisionMeasurement | None) -> None:
        if meas is None:
            return
        lat, vert = meas.position_sigma()
        sigma = self.sigma_inflation * np.array([lat, lat, vert])
        if not self.initialised or self.rejects >= self.reinit_after:
            self.kf.init(meas.p, sigma)
            self.att.init(meas.R_wc)
            self.initialised, self.age, self.rejects = True, 0, 0
            return
        ok, _ = self.kf.update(meas.p, sigma)
        if ok:
            gain = float(np.clip(0.06 + 0.002 * meas.width_px, 0.08, 0.6))
            self.att.update(meas.R_wc, gain, self.dt)
            self.age, self.rejects = 0, 0
        else:
            self.rejects += 1

    def step(
        self, f_b: np.ndarray, w_b: np.ndarray, meas: VisionMeasurement | None
    ) -> Estimate | None:
        self.predict(f_b, w_b)
        self.update(meas)
        return self.estimate()

    def estimate(self) -> Estimate | None:
        if not self.initialised:
            return None
        return Estimate(
            p=self.kf.x[:3].copy(),
            v=self.kf.x[3:6].copy(),
            R_wc=self.att.R.copy(),
            pos_std=self.kf.pos_std,
            age=self.age,
            lost_time=self.age * self.dt,
        )
