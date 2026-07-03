#!/usr/bin/env python3

import cv2
import numpy as np
import pyrealsense2 as rs
import os
import json

# ============================================================
# Checkerboard settings
# ============================================================

CHECKERBOARD = (4, 3)
SQUARE_SIZE = 0.030

JSON_DIR = "eye_data_json"
IMAGE_DIR = "eye_data_images"

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ============================================================
# RealSense setup
# ============================================================

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

camera_matrix = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64).reshape(-1, 1)

print("\nCamera Matrix:\n", camera_matrix)
print("\nDistortion:\n", dist_coeffs.flatten())

# ============================================================
# Object points (CENTER origin)
# ============================================================

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

grid = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2).astype(np.float32)

grid[:, 0] -= (CHECKERBOARD[0] - 1) / 2.0
grid[:, 1] -= (CHECKERBOARD[1] - 1) / 2.0

objp[:, :2] = grid * SQUARE_SIZE

# ============================================================

existing = []
for f in os.listdir(JSON_DIR):
    if f.startswith("eye_") and f.endswith(".json"):
        try:
            existing.append(int(f.split("_")[1].split(".")[0]))
        except:
            pass

sample_id = 0 if len(existing) == 0 else max(existing) + 1

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

print("\n================================")
print("Press s   -> Save sample")
print("Press ESC -> Exit")
print("================================")

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    image = np.asanyarray(color_frame.get_data())
    display = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FAST_CHECK
    )

    pose_valid = False

    if found:

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)

        success, rvec, tvec = cv2.solvePnP(
            objp,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:

            pose_valid = True

            R, _ = cv2.Rodrigues(rvec)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()

            x, y, z = tvec.flatten()

            # ============================================================
            # 🔥 THIS IS THE ONLY ADDITION: DIRECTION AXES
            # ============================================================
            cv2.drawFrameAxes(
                display,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                SQUARE_SIZE
            )

            cv2.putText(
                display,
                f"Center X = {x:.3f} m",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Center Y = {y:.3f} m",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Center Z = {z:.3f} m",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Samples: {sample_id}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

    cv2.imshow("Eye-to-Hand Capture", display)

    key = cv2.waitKey(1) & 0xFF

    # ============================================================
    # SAVE (UNCHANGED)
    # ============================================================

    if key == ord('s'):

        if not pose_valid:
            print("Checkerboard not detected.")
            continue

        data = {
            "sample_id": sample_id,
            "R": R.tolist(),
            "t": tvec.tolist(),
            "T": T.tolist(),
            "rvec": rvec.tolist()
        }

        json_name = os.path.join(JSON_DIR, f"eye_{sample_id:03d}.json")

        with open(json_name, "w") as f:
            json.dump(data, f, indent=4)

        cv2.imwrite(
            os.path.join(IMAGE_DIR, f"eye_{sample_id:03d}_raw.png"),
            image
        )

        cv2.imwrite(
            os.path.join(IMAGE_DIR, f"eye_{sample_id:03d}_annotated.png"),
            display
        )

        print(
            f"Saved Sample {sample_id} "
            f"(CENTER) -> "
            f"X={x:.4f} "
            f"Y={y:.4f} "
            f"Z={z:.4f}"
        )

        sample_id += 1

    elif key == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()
