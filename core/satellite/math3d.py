"""Minimal 3D math helpers (numpy) for the free-floating satellite simulation.

Quaternions are Hamilton convention ``(w, x, y, z)`` and represent the rotation that maps
*body* coordinates into *world* coordinates.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------- quaternions ----
def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else quat_identity()


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return quat_identity()
    axis = axis / n
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Minimal rotation vector (axis * angle) of a unit quaternion."""
    q = quat_normalize(q)
    if q[0] < 0:  # canonical hemisphere -> angle in [0, pi]
        q = -q
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return q[1:] / s * angle


def rotvec_quat(rv: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, dtype=np.float64)
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return quat_identity()
    return quat_from_axis_angle(rv / angle, angle)


def quat_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q``."""
    return quat_matrix(q) @ np.asarray(v, dtype=np.float64)


def quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two orientations, in radians."""
    qa = quat_normalize(a)
    qb = quat_normalize(b)
    d = abs(float(np.dot(qa, qb)))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def integrate_quat(q: np.ndarray, omega_world: np.ndarray, dt: float) -> np.ndarray:
    """First order integration of ``q`` under a world-frame angular velocity."""
    w = omega_world * dt
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        return quat_normalize(q)
    dq = quat_from_axis_angle(w / angle, angle)
    return quat_normalize(quat_multiply(dq, q))


# --------------------------------------------------------------------- misc ----
def unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


def smoothstep(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return float(t * t * (3 - 2 * t))


def approach(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    """Move ``current`` towards ``target`` by at most ``max_step`` (euclidean)."""
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = target - current
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist < 1e-12:
        return target.copy()
    return current + delta / dist * max_step
