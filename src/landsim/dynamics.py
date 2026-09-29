"""Drone truth dynamics, wind, attitude motion and IMU simulation.

The inner loop is modelled as a first-order velocity tracker with time constant
``tau``; wind enters as an acceleration disturbance (steady part plus Ornstein-Uhlenbeck
gusts). Roll/pitch combine a tilt coupled to commanded acceleration with oscillation.
The camera is rigidly mounted at the drone reference point, looking down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import GRAVITY, attitude_from_rpy, log_so3
from .nuisance import Nuisance


@dataclass
class ImuSample:
    f_b: np.ndarray  # specific force in the body/camera frame (m/s^2)
    w_b: np.ndarray  # angular rate in the body/camera frame (rad/s)


@dataclass
class Scenario:
    """Everything random about one run, drawn once from the seed."""

    seed: int
    level: float
    p0: np.ndarray
    yaw0: float
    yaw_rate: float
    wind_mean: np.ndarray
    gust_sigma: float
    tilt_amp: np.ndarray
    tilt_freq: np.ndarray
    tilt_phase: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    nuisance: Nuisance
    accel_noise: float = 0.05
    gyro_noise: float = 0.004
    tau: float = 0.4
    dt: float = 1 / 30
    rng_streams: dict = field(default_factory=dict, repr=False)

    @classmethod
    def sample(cls, seed: int, level: float = 0.5) -> Scenario:
        ss = np.random.SeedSequence(seed)
        r_scn, r_wind, r_imu, r_img = (np.random.default_rng(s) for s in ss.spawn(4))
        ang, rad = r_scn.uniform(0, 2 * np.pi), r_scn.uniform(0.5, 3.0)
        p0 = np.array([rad * np.cos(ang), rad * np.sin(ang), r_scn.uniform(9.0, 11.0)])
        wang, wmag = r_scn.uniform(0, 2 * np.pi), r_scn.uniform(0.0, 0.25)
        scn = cls(
            seed=seed,
            level=level,
            p0=p0,
            yaw0=float(r_scn.uniform(-0.7, 0.7)),
            yaw_rate=float(r_scn.uniform(-0.05, 0.05)),
            wind_mean=np.array([wmag * np.cos(wang), wmag * np.sin(wang)]),
            gust_sigma=float(r_scn.uniform(0.05, 0.2)),
            tilt_amp=np.deg2rad(r_scn.uniform(2.0, 6.0, 2)),
            tilt_freq=r_scn.uniform(0.3, 0.8, 2),
            tilt_phase=r_scn.uniform(0, 2 * np.pi, 2),
            accel_bias=r_imu.normal(0, 0.05, 3),
            gyro_bias=r_imu.normal(0, 0.003, 3),
            nuisance=Nuisance.sample(level, r_scn),
        )
        scn.rng_streams = {"wind": r_wind, "imu": r_imu, "img": r_img}
        return scn


class World:
    """Steps the true state of the drone and produces IMU readings."""

    z_touchdown = 0.15

    def __init__(self, scn: Scenario):
        self.scn = scn
        self.dt = scn.dt
        self.t = 0.0
        self.p = scn.p0.copy()
        self.v = np.zeros(3)
        self.gust = np.zeros(2)
        self.tilt_state = np.zeros(2)  # low-passed acceleration-coupled roll, pitch
        self.rng_wind = scn.rng_streams["wind"]
        self.rng_imu = scn.rng_streams["imu"]
        self.R = self._attitude(0.0)
        self.omega_b = np.zeros(3)
        self.touched_down = False
        self.touchdown_p: np.ndarray | None = None

    def _attitude(self, t: float) -> np.ndarray:
        s = self.scn
        osc = s.tilt_amp * np.sin(2 * np.pi * s.tilt_freq * t + s.tilt_phase)
        osc += 0.4 * s.tilt_amp * np.sin(2 * np.pi * 2.3 * s.tilt_freq * t + 1.7 * s.tilt_phase)
        roll = self.tilt_state[0] + osc[0]
        pitch = self.tilt_state[1] + osc[1]
        yaw = s.yaw0 + s.yaw_rate * t + 0.03 * np.sin(2 * np.pi * 0.2 * t)
        return attitude_from_rpy(roll, pitch, yaw)

    def step(self, v_cmd: np.ndarray) -> ImuSample:
        s, dt = self.scn, self.dt
        a_ctrl = np.clip((v_cmd - self.v) / s.tau, -3.0, 3.0)
        theta = 0.5  # gust OU correlation rate (1/s)
        self.gust += -theta * self.gust * dt + s.gust_sigma * np.sqrt(
            2 * theta * dt
        ) * self.rng_wind.standard_normal(2)
        a = a_ctrl.copy()
        a[:2] += s.wind_mean + self.gust
        a[2] += 0.02 * self.rng_wind.standard_normal()

        p_prev, z_prev = self.p.copy(), self.p[2]
        p_new = self.p + self.v * dt + 0.5 * a * dt**2
        self.v = self.v + a * dt

        # tilt coupled to (low-passed) commanded lateral acceleration
        target = np.array([-a_ctrl[1], a_ctrl[0]]) / GRAVITY
        self.tilt_state += (target - self.tilt_state) * (dt / 0.15)
        R_prev = self.R
        self.t += dt
        self.R = self._attitude(self.t)
        self.omega_b = log_so3(R_prev.T @ self.R) / dt

        if p_new[2] <= self.z_touchdown and not self.touched_down:
            f = (z_prev - self.z_touchdown) / max(z_prev - p_new[2], 1e-9)
            self.touchdown_p = p_prev + f * (p_new - p_prev)
            self.touched_down = True
        self.p = p_new

        f_w = a + np.array([0.0, 0.0, GRAVITY])
        f_b = self.R.T @ f_w + s.accel_bias + self.rng_imu.normal(0, s.accel_noise, 3)
        w_b = self.omega_b + s.gyro_bias + self.rng_imu.normal(0, s.gyro_noise, 3)
        return ImuSample(f_b, w_b)
