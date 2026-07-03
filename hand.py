#!/usr/bin/env python3
# coding=utf-8
"""

Saved JSON format (identical to original hand.py so solver.py is unchanged):
  {
    "sample_id": N,
    "read_joints": [j1..j6],   # degrees, from robot firmware
    "R": [[3x3]],               # rotation matrix, gripper-in-base
    "t": [[x],[y],[z]]          # translation in METERS, gripper-in-base
  }

Usage (same as original hand.py):
  f + ENTER  →  Free Drive  (relax motors, move arm by hand)
  h + ENTER  →  Hold        (lock motors)
  s + ENTER  →  Save        (compute FK from joints, save JSON)
  q + ENTER  →  Quit
"""

import os
import sys
import json
import time
import numpy as np
from pymycobot.mycobot import MyCobot

# ── Custom IK solver imports ───────────────────────────────────────────────────
# Ensure this script is run from the myIKsolver project directory so that
# kinematics.py, config.py, parser.py, etc. are on the path.
from kinematics import forward_kinematics_tool


# ============================================================
# CONFIGURATION
# ============================================================

SAVE_DIR = "hand_data"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# HELPER: LOCK / HOLD POSITION
# ============================================================

def hold_position(robot_obj):
    """Re-engages all joint motors to lock the arm in its current pose."""
    robot_obj.power_on()
    time.sleep(0.1)
    for idx in range(1, 7):
        try:
            robot_obj.focus_servo(idx)
            time.sleep(0.05)
        except AttributeError:
            pass
    time.sleep(0.1)


# ============================================================
# HELPER: READ JOINT ANGLES (RADIANS) FROM ROBOT
# ============================================================

def read_joint_angles_rad(robot_obj, retries=5):
    """
    Read joint angles from the robot firmware.

    Returns
    -------
    angles_deg : list[float]  – raw degrees from firmware (logged only)
    angles_rad : np.ndarray   – same angles in radians, fed to FK
    None, None on failure.
    """
    for attempt in range(retries):
        raw = robot_obj.get_angles()

        if isinstance(raw, (list, tuple)) and len(raw) == 6:
            angles_deg = [float(a) for a in raw]
            angles_rad = np.radians(angles_deg)
            return angles_deg, angles_rad

        print(f"  ...Joint read returned unexpected value. "
              f"Attempt {attempt + 1}/{retries}. Got: {raw}")
        time.sleep(0.2)

    return None, None


# ============================================================
# MAIN
# ============================================================

print("Connecting to JetCobot...")
mc = MyCobot('/dev/ttyUSB0', 1000000)
time.sleep(1)

print("Initializing motor states...")
hold_position(mc)

print("============================================================")
print("EYE-TO-HAND CALIBRATION  –  IK Solver FK Mode")
print()
print("  Pose is computed via forward_kinematics_tool() from your")
print("  URDF model instead of mc.get_coords() Euler angles.")
print()
print("  f + ENTER  →  Free Drive  (relax arm, move by hand)")
print("  h + ENTER  →  Hold        (lock motors in place)")
print("  s + ENTER  →  Save        (FK pose → JSON)")
print("  q + ENTER  →  Quit")
print("============================================================")

# Find the next available sample ID
existing_ids = []
for fname in os.listdir(SAVE_DIR):
    if fname.startswith("hand_") and fname.endswith(".json"):
        try:
            existing_ids.append(int(fname.split("_")[1].split(".")[0]))
        except (IndexError, ValueError):
            pass

sample_id = 0 if not existing_ids else max(existing_ids) + 1
print(f"\nStarting at sample index: {sample_id}\n")


while True:
    print(f"--- [Saved so far: {sample_id}] ---")
    cmd = input("Command (f / h / s / q): ").strip().lower()

    # ── Quit ──────────────────────────────────────────────────────────────────
    if cmd == 'q':
        print("\nRe-engaging motors for safety...")
        hold_position(mc)
        break

    # ── Free Drive ────────────────────────────────────────────────────────────
    elif cmd == 'f':
        print("\n[FREE DRIVE] Motors relaxed. Move arm manually...")
        mc.release_all_servos()

    # ── Hold Position ─────────────────────────────────────────────────────────
    elif cmd == 'h':
        print("\n[HOLD] Locking motors in place.")
        hold_position(mc)

    # ── Save Sample ───────────────────────────────────────────────────────────
    elif cmd == 's':
        print("\n[SAVE] Reading joint angles from firmware...")
        time.sleep(0.5)  # Let serial line settle

        angles_deg, angles_rad = read_joint_angles_rad(mc)

        if angles_rad is None:
            print("[ERROR] Could not read joint angles. Try again.")
            continue

        # ── Compute FK via custom IK solver ───────────────────────────────────
        #
        # forward_kinematics_tool() returns a 4x4 homogeneous matrix:
        #
        #   T = | R  t |     R : 3×3 rotation  (gripper frame in base frame)
        #       | 0  1 |     t : 3×1 translation in METERS
        #
        # This is exactly "Gripper-in-Base" (T_base_to_gripper), which is
        # what solver.py expects in hand_data/*.json.
        #
        T = forward_kinematics_tool(angles_rad)

        R = T[:3, :3]           # 3×3 rotation matrix
        t = T[:3, 3].reshape(3, 1)  # 3×1 translation in metres

        # ── Sanity check: R must be a valid rotation matrix ───────────────────
        det = np.linalg.det(R)
        ortho_err = np.linalg.norm(R @ R.T - np.eye(3))

        if abs(det - 1.0) > 0.01 or ortho_err > 0.01:
            print(f"[WARNING] FK rotation matrix looks invalid! "
                  f"det={det:.4f}, ortho_err={ortho_err:.4f}")
            print("  Check URDF / parser. Saving anyway but verify result.")

        # ── Build and save JSON ───────────────────────────────────────────────
        data = {
            "sample_id": sample_id,
            "read_joints": angles_deg,      # degrees, for human reference
            "R": R.tolist(),                # 3×3 rotation matrix
            "t": t.tolist()                 # [[x],[y],[z]] in metres
        }

        filename = os.path.join(SAVE_DIR, f"hand_{sample_id:03d}.json")

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

        # ── Pretty-print what was saved ───────────────────────────────────────
        x_mm, y_mm, z_mm = t.flatten() * 1000.0

        print(f"[OK] Saved: {filename}")
        print(f"     Joints (deg) : {[f'{a:.2f}' for a in angles_deg]}")
        print(f"     Position (mm): X={x_mm:.2f}  Y={y_mm:.2f}  Z={z_mm:.2f}")
        print(f"     FK det(R)    : {det:.6f}  (should be 1.0)")

        sample_id += 1

    elif cmd == '':
        continue

    else:
        print(f"[INVALID] '{cmd}' not recognised. Use f, h, s, or q.")


print("\n============================================================")
print(f"Session complete. {sample_id} samples saved in '{SAVE_DIR}/'.")
print("Run solver.py next to compute T_cam2base.npy")
print("============================================================")
