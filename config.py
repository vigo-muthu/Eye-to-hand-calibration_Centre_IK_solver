import numpy as np

# ── Robot model ────────────────────────────────────────────────────────────────
JOINT_COUNT = 6
URDF_FILE   = "jetcobot.urdf"

# ── IK solver ──────────────────────────────────────────────────────────────────
POSITION_TOLERANCE_M = 0.001        # 1 mm
MAX_ITERATIONS       = 200
DAMPING              = 0.01

# ── Tool offset (measured with test_tool_offset.py) ───────────────────────────
TOOL_OFFSET_LOCAL_MM = np.array([16.8112, 2.2664, -4.2051], dtype=float)

# ── main.py correction loop ───────────────────────────────────────────────────
MAX_CORRECTION_ATTEMPTS = 4
REAL_ERROR_LIMIT_MM     = 2.0
CORRECTION_GAIN         = 0.7
MAX_PLANNED_ERROR_MM    = 2.0

# ── Robot communication ────────────────────────────────────────────────────────
PORT = "/dev/ttyUSB0"
BAUD = 1000000

# ── Workspace hard limits (used by range check) ───────────────────────────────
WORKSPACE_X_MM = (-280.0, 280.0)
WORKSPACE_Y_MM = (-280.0, 280.0)
WORKSPACE_Z_MM = (  50.0, 280.0)
