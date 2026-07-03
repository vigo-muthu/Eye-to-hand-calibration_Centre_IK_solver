"""
ik_solver.py  –  Position-only IK with multi-guess restart.

Design
------
inverse_kinematics() tries several fixed seed postures plus the current
robot pose.  Each seed gets MAX_ITERATIONS of DLS gradient descent via
solve_once().  The best solution is chosen by a combined score:

    score = position_error_mm + POSTURE_WEIGHT * joint_movement_degrees

POSTURE_WEIGHT = 0.05 means we accept up to 1 mm extra position error to
save 20 degrees of joint movement.  Tune this in config.py if needed.

solve_once() improvements over the original:
  - Soft joint-limit gradient instead of hard mid-step clamp
    (prevents oscillation at limits)
  - Hard clamp only at the very end
"""

import numpy as np
from config import JOINT_COUNT, MAX_ITERATIONS, DAMPING, POSITION_TOLERANCE_M
from kinematics import forward_kinematics_tool
from jacobian   import compute_jacobian
from parser     import load_robot_model


POSTURE_WEIGHT = 0.05   # mm per degree – how much we penalise joint movement


# ── Joint utilities ────────────────────────────────────────────────────────────

def _load_joints():
    return load_robot_model()[:JOINT_COUNT]


def clamp_to_limits(angles, joints):
    angles = angles.copy()
    for i, joint in enumerate(joints):
        if joint.lower is not None:
            angles[i] = max(angles[i], joint.lower)
        if joint.upper is not None:
            angles[i] = min(angles[i], joint.upper)
    return angles


def _joint_limit_gradient(angles, joints, weight=0.001):
    """Soft gradient that repels joints from their limits."""
    grad = np.zeros(len(angles))
    for i, joint in enumerate(joints):
        if joint.lower is not None and joint.upper is not None:
            mid  = (joint.lower + joint.upper) / 2.0
            rang = (joint.upper - joint.lower) / 2.0
            if rang > 0:
                grad[i] = weight * (angles[i] - mid) / (rang ** 2)
    return grad


# ── Single DLS solve ───────────────────────────────────────────────────────────

def solve_once(target_position, initial_angles, joints):
    """
    Run DLS gradient descent from one starting posture.

    Returns
    -------
    best_angles : np.ndarray (6,)
    best_error  : float  (metres)
    iterations  : int
    """
    target_position = np.array(target_position, dtype=float)
    angles          = np.array(initial_angles,  dtype=float)

    best_angles = angles.copy()
    best_error  = float("inf")

    for iteration in range(MAX_ITERATIONS):
        current_position = forward_kinematics_tool(angles)[:3, 3]
        error            = target_position - current_position
        error_norm       = np.linalg.norm(error)

        if error_norm < best_error:
            best_error  = error_norm
            best_angles = angles.copy()

        if error_norm < POSITION_TOLERANCE_M:
            return angles, error_norm, iteration

        # DLS pseudo-inverse on position rows only
        J          = compute_jacobian(angles)
        J_pos      = J[:3, :]
        J_pinv     = J_pos.T @ np.linalg.inv(
            J_pos @ J_pos.T + (DAMPING ** 2) * np.eye(3)
        )

        delta      = J_pinv @ error
        delta     -= _joint_limit_gradient(angles, joints)   # soft limit repulsion

        max_step   = np.radians(5)
        delta      = np.clip(delta, -max_step, max_step)

        angles = angles + delta
        # No hard clamp mid-solve; it fights the gradient

    # Hard clamp only at the end
    return clamp_to_limits(best_angles, joints), best_error, MAX_ITERATIONS


# ── Public entry point ─────────────────────────────────────────────────────────

def inverse_kinematics(target_position, initial_angles=None):
    """
    Solve position-only IK.

    Parameters
    ----------
    target_position : array-like (3,)  – target tool position in metres
    initial_angles  : array-like (6,)  – current robot angles in radians.
                      Used as the first seed and as the posture reference.

    Returns
    -------
    angles : np.ndarray (6,)  – solution in radians
    """
    joints = _load_joints()

    if initial_angles is not None:
        current = np.array(initial_angles, dtype=float)
    else:
        current = np.zeros(JOINT_COUNT)

    # Seeds: current pose first, then diverse fixed postures
    seeds = [
        current,
        np.radians([  0,  30, -30,  30,  0, 0]),
        np.radians([  0,  20, -60,  40,  0, 0]),
        np.radians([  0, -20, -60,  60,  0, 0]),
        np.radians([ 30,  20, -60,  40,  0, 0]),
        np.radians([-30,  20, -60,  40,  0, 0]),
    ]

    best_angles     = None
    best_error      = float("inf")
    best_score      = float("inf")
    best_iterations = MAX_ITERATIONS

    for seed in seeds:
        angles, error, iterations = solve_once(target_position, seed, joints)

        # Penalise big jumps from the current pose
        movement_deg = float(np.linalg.norm(np.degrees(angles - current)))
        score        = (error * 1000.0) + POSTURE_WEIGHT * movement_deg

        if score < best_score:
            best_score      = score
            best_error      = error
            best_angles     = angles
            best_iterations = iterations

    if best_error < POSITION_TOLERANCE_M:
        print(f"IK solved in {best_iterations} iterations")
    else:
        print(f"IK best error mm: {best_error * 1000:.3f}")

    return best_angles
