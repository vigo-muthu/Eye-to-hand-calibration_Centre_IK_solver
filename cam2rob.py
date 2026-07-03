#!/usr/bin/env python3
import numpy as np
import os

# 1. Load the calibration matrix
matrix_file = "T_cam2base.npy"

if not os.path.exists(matrix_file):
    print(f"Error: {matrix_file} not found in this folder!")
    exit()

T_cam2base = np.load(matrix_file)
print("Calibration Matrix Loaded:")
print(T_cam2base)

# 2. Interactive Loop
while True:
    print("\n------------------------------------------------")
    print("Enter camera coordinates (in meters) or 'q' to quit:")
    
    x = input("Cam X: ")
    if x.lower() == 'q': break
    y = input("Cam Y: ")
    if y.lower() == 'q': break
    z = input("Cam Z: ")
    if z.lower() == 'q': break
    
    try:
        # Create 3D vector from input
        cam_pos = np.array([float(x), float(y), float(z)])
        
        # Convert to a 4D homogeneous coordinate [X, Y, Z, 1.0]
        cam_h = np.append(cam_pos, 1.0)
        
        # Matrix multiplication: Base = T * Camera
        base_h = T_cam2base @ cam_h
        base_pos = base_h[:3]
        
        # Print results in both meters and millimeters
        print("\n--- TRANSFORMATION RESULT ---")
        print(f"Camera Frame (m):  [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]")
        print(f"Robot Base (m):    [{base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f}]")
        print(f"Robot Base (mm):   [{base_pos[0]*1000:.1f}, {base_pos[1]*1000:.1f}, {base_pos[2]*1000:.1f}]")
        
    except ValueError:
        print("Invalid input! Please enter numbers only.")
