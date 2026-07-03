import numpy as np
import time
from pymycobot.mycobot import MyCobot

from ik_solver import inverse_kinematics   # ✔ correct function


# ==========================================================
# ROBOT INIT
# ==========================================================
mc = MyCobot('/dev/ttyUSB0', 1000000)
mc.power_on()
time.sleep(2)


# ==========================================================
# MOVE FUNCTION
# ==========================================================
def move_to_xyz(x, y, z):

    # ------------------------------------------------------
    # 1. Convert mm → meters (VERY IMPORTANT)
    # ------------------------------------------------------
    target = np.array([x, y, z], dtype=float) / 1000.0

    # ------------------------------------------------------
    # 2. Get current joint state (for smooth IK)
    # ------------------------------------------------------
    current_angles = mc.get_angles()

    if current_angles is None:
        print("Robot not responding")
        return

    current_angles = np.radians(current_angles)

    # ------------------------------------------------------
    # 3. Solve IK
    # ------------------------------------------------------
    joint_angles = inverse_kinematics(
        target_position=target,
        initial_angles=current_angles
    )

    if joint_angles is None:
        print("IK failed")
        return

    # ------------------------------------------------------
    # 4. Convert radians → degrees (robot requirement)
    # ------------------------------------------------------
    joint_angles_deg = np.degrees(joint_angles)

    print("Sending joint angles:", joint_angles_deg)

    # ------------------------------------------------------
    # 5. Send to robot
    # ------------------------------------------------------
    mc.send_angles(joint_angles_deg.tolist(), 20)
    time.sleep(2)


# ==========================================================
# USER INTERFACE LOOP
# ==========================================================
if __name__ == "__main__":

    while True:

        try:
            user_input = input("\nEnter X Y Z in mm (or q): ")

            if user_input.lower() == "q":
                break

            x, y, z = map(float, user_input.split())

            move_to_xyz(x, y, z)

        except Exception as e:
            print("Error:", e)
