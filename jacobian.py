import numpy as np
from kinematics import forward_kinematics
from config import JOINT_COUNT


def compute_jacobian(joint_angles):

    T, frames = forward_kinematics(joint_angles, return_frames=True)

    end_pos = T[:3, 3]

    J = np.zeros((6, JOINT_COUNT))

    for i, (joint, joint_origin_T, _) in enumerate(frames):

        joint_pos = joint_origin_T[:3, 3]

        # CRITICAL FIX: DO NOT rotate axis by frame
        # treat axis as WORLD-fixed from model definition

        axis_world = np.array(joint.axis, dtype=float)
        axis_world = axis_world / np.linalg.norm(axis_world)

        r = end_pos - joint_pos

        J[0:3, i] = np.cross(axis_world, r)
        J[3:6, i] = axis_world

    return J
