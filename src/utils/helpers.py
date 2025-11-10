import yaml
import numpy as np
from pathlib import Path


def setup_simulation(cfg):
    """Prepares all static simulation data."""
    # Paths
    base_dir = Path.cwd()
    datapath = str(base_dir / cfg['paths']['dataset_dir'] / cfg['paths']['ply_filename'])
    output_path = str(base_dir / cfg['paths']['result_dir'] / cfg['paths']['camera_mcap_filename'])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # World Setup
    start_pos_world = np.array(cfg['world']['start_pos'])
    utm_offset = np.array(cfg['world']['utm_offset'])
    
    # Camera's start position relative to the shifted origin
    start_pos_relative = start_pos_world - utm_offset

    # Camera Rotation
    # 1. Pitch rotation (looking down/forward)
    R_pitch = np.array([
        [1,  0,  0],
        [0,  0,  -1],
        [0,  1,  0]
    ])
    
    # 2. Yaw rotation (steering left/right)
    theta = np.radians(cfg['camera']['yaw_offset_deg'])
    c, s = np.cos(theta), np.sin(theta)
    R_yaw = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    # Combine rotations: Yaw first, then Pitch
    R_world_to_cam = R_pitch @ R_yaw

    # Motion Vector
    # We need the camera-to-world matrix to find the "forward" direction in world coordinates
    R_cam_to_world = R_world_to_cam.T
    motion_dir = R_cam_to_world[:, 2] # Forward vector is Z-axis of camera in world
    motion_vec = motion_dir / np.linalg.norm(motion_dir)

    # Camera Intrinsics
    W, H = cfg['camera']['width'], cfg['camera']['height']
    FL = cfg['camera']['focal_length_px']
    K = np.array([
        [FL, 0, W / 2.0],
        [0, FL, H / 2.0],
        [0,  0,       1]
    ])

    return datapath, output_path, start_pos_relative, R_world_to_cam, motion_vec, K

def get_pose(frame_idx, total_frames, start_pos, motion_vec, speed_mps, fps):
    """Calculates the camera position for a given frame based on speed."""
    # 1. Calculate time elapsed
    time_elapsed = frame_idx / fps
    
    # 2. Calculate distance traveled
    distance = speed_mps * time_elapsed
    
    # 3. Calculate current position
    current_pos = start_pos + (motion_vec * distance)
    
    return current_pos