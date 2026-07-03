import math
import numpy as np
from parser import load_robot_model
from config import JOINT_COUNT, TOOL_OFFSET_LOCAL_MM


# ==========================================================
# ROTATIONS
# ==========================================================

def rotation_x(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

def rotation_y(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rotation_z(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


# ==========================================================
# TRANSFORMS
# ==========================================================

def homogeneous_transform(xyz, rpy):
    roll, pitch, yaw = rpy

    T = np.eye(4)
    T[:3, :3] = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)
    T[:3, 3] = np.array(xyz, dtype=float)
    return T


def axis_angle_rotation(axis, angle):
    axis = np.array(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-9)

    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)

    return np.array([
        [c + x*x*(1-c),     x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s,   c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s,   z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])


# ==========================================================
# FORWARD KINEMATICS (CORE)
# ==========================================================

def forward_kinematics(joint_angles, return_frames=False):
    joints = load_robot_model()[:JOINT_COUNT]

    T = np.eye(4)
    frames = []

    for joint, angle in zip(joints, joint_angles):

        # joint fixed transform
        T = T @ homogeneous_transform(joint.xyz, joint.rpy)

        joint_origin = T.copy()

        # joint rotation
        R = np.eye(4)
        R[:3, :3] = axis_angle_rotation(joint.axis, angle)

        T = T @ R

        frames.append((joint, joint_origin.copy(), T.copy()))

    if return_frames:
        return T, frames

    return T


# ==========================================================
# TOOL FK (FIXED - SINGLE CONSISTENT DEFINITION)
# ==========================================================

def forward_kinematics_tool(joint_angles):
    """
    Tool frame = FK + fixed offset in end-effector frame.
    """

    T = forward_kinematics(joint_angles)

    tool_offset_m = np.array(TOOL_OFFSET_LOCAL_MM, dtype=float) / 1000.0

    T_tool = T.copy()
    T_tool[:3, 3] = T[:3, 3] + T[:3, :3] @ tool_offset_m

    return T_tool
