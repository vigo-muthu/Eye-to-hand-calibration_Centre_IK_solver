#!/usr/bin/env python3
# coding=utf-8

import cv2
import numpy as np
import pyrealsense2 as rs

# Global variable to store the latest mouse click coordinates
clicked_pixel = None

# ============================================================
# MOUSE CALLBACK FUNCTION
# ============================================================
def mouse_callback(event, x, y, flags, param):
    """Captures the (x, y) pixel coordinates when the left mouse button is clicked."""
    global clicked_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (x, y)
        print(f"Clicked pixel: ({x}, {y})")

# ============================================================
# INITIALIZATION (REALSENSE PIPELINE)
# ============================================================
print("Starting RealSense D415 pipeline...")
pipeline = rs.pipeline()
config = rs.config()

# Enable BOTH Color and Depth streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Get camera intrinsics (needed for 2D to 3D deprojection)
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()

# Create an alignment object.
# This ensures that depth pixels line up perfectly with color pixels.
align_to = rs.stream.color
align = rs.align(align_to)

# Setup OpenCV Window and attach the mouse click callback
window_name = "Live Stream - Click for 3D Coordinates"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

print("\n============================================================")
print("INSTRUCTIONS:")
print("  - Left Click anywhere on the video stream to get X, Y, Z coordinates.")
print("  - Press ESC to exit the program.")
print("============================================================\n")

# ============================================================
# MAIN LOOP
# ============================================================
try:
    while True:
        # Wait for the next set of frames
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Extract both aligned frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # Validate that both frames are available
        if not color_frame or not depth_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        display_image = color_image.copy()

        # If the user has clicked on the screen, calculate and display coordinates
        if clicked_pixel is not None:
            px, py = clicked_pixel
            
            # 1. Get the depth (in meters) for the specific clicked pixel
            depth_in_meters = depth_frame.get_distance(px, py)

            if depth_in_meters > 0:
                # 2. Convert 2D pixel + Depth to 3D point (Deprojection)
                point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth_in_meters)
                x, y, z = point_3d[0], point_3d[1], point_3d[2]

                # 3. Draw a circle where the user clicked
                cv2.circle(display_image, (px, py), 5, (0, 0, 255), -1)

                # 4. Display the coordinates on the screen next to the point
                text = f"X: {x:.3f}m Y: {y:.3f}m Z: {z:.3f}m"
                
                # Ensure text doesn't draw off-screen if clicked near the edge
                text_x = max(10, min(px + 10, 640 - 250))
                text_y = max(20, min(py - 10, 480 - 10))
                
                cv2.putText(display_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # If depth is 0, the sensor couldn't get a reading there (shiny/black surface or out of range)
                cv2.circle(display_image, (px, py), 5, (0, 0, 255), -1)
                cv2.putText(display_image, "Invalid Depth Info", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Show the image
        cv2.imshow(window_name, display_image)

        # Exit on ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Exiting...")
            break

finally:
    # Cleanup safely
    pipeline.stop()
    cv2.destroyAllWindows()
