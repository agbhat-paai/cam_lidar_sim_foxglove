import numpy as np
from pathlib import Path
import json
from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding

def setup_simulation(cfg, sensor_type='camera'):
    """Prepares all static simulation data."""
    base_dir = Path.cwd()
    datapath = str(base_dir / cfg["paths"]["dataset_dir"] / cfg["paths"]["ply_filename"])
    if sensor_type == 'camera':
        output_path = str(base_dir / cfg["paths"]["result_dir"] / cfg["paths"]["camera_mcap_filename"])
    else:
        output_path = str(base_dir / cfg["paths"]["result_dir"] / cfg["paths"]["lidar_mcap_filename"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # World Setup
    start_pos_world = np.array(cfg["world"]["start_pos"])
    utm_offset = np.array(cfg["world"]["utm_offset"])

    # Camera's start position relative to the shifted origin
    start_pos_relative = start_pos_world - utm_offset

    # Camera Rotation
    # Default R
    R_pitch = np.array([[1, 0, 0], 
                        [0, 0, -1], 
                        [0, 1, 0]])

    # Adding small Yaw (trial and error correction)
    theta = np.radians(cfg["camera"]["yaw_offset_deg"]) + np.pi
    c, s = np.cos(theta), np.sin(theta)
    R_yaw = np.array([[c, -s, 0], 
                      [s, c, 0], 
                      [0, 0, 1]])

    # final rotation
    R_world_to_cam = R_pitch @ R_yaw

    # motion Vector
    R_cam_to_world = R_world_to_cam.T
    motion_dir = R_cam_to_world[:, 2]  # forward direction in world frame

    #fix drift
    motion_dir[2] = motion_dir[2] - 0.02  # eliminating vertical drift, need a better solution

    motion_vec = motion_dir / np.linalg.norm(motion_dir)  # normalize

    # camera intrinsics
    W, H = cfg["camera"]["width"], cfg["camera"]["height"]
    FL = W / 2.0
    K = np.array([[FL, 0, W / 2.0], 
                  [0, FL, H / 2.0], 
                  [0, 0, 1]])

    return datapath, output_path, start_pos_relative, R_world_to_cam, motion_vec, K



def get_pose(frame_idx, total_frames, start_pos, motion_vec, speed_mps, fps):
    """Calculates the camera position for a given frame based on speed."""
    time_elapsed = frame_idx / fps  # time elapsed
    distance = speed_mps * time_elapsed  # distance traveled
    current_pos = start_pos + (motion_vec * distance)  # curr position

    return current_pos


def setup_foxglove_lidar_schema(writer):
    """
    Registers the foxglove.PointCloud schema for (x,y,z,r,g,b) data.
    """
    # We will pack our data as:
    # x: float32 (4 bytes)
    # y: float32 (4 bytes)
    # z: float32 (4 bytes)
    # r: uint8   (1 byte)
    # g: uint8   (1 byte)
    # b: uint8   (1 byte)
    # padding: 1 byte
    # TOTAL STRIDE: 16 bytes
    
    schema_json = {
        "type": "object",
        "properties": {
            "timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
            "frame_id": {"type": "string"},
            "pose": {"type": "object", "properties": {"position": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}}, "orientation": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}, "w": {"type": "number"}}}}},
            "point_stride": {"type": "integer", "description": "Number of bytes per point"},
            "fields": {
                "type": "array", 
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "offset": {"type": "integer"},
                        "type": {"type": "integer", "description": "Numeric type (1=uint8, 7=float32)"}
                    }
                }
            },
            "data": {"type": "string", "contentEncoding": "base64"}
        }
    }

    schema_id = writer.register_schema(
        name="PointCloud",
        encoding=SchemaEncoding.JSONSchema,
        data=json.dumps(schema_json).encode("utf-8"),
    )
    
    channel_id = writer.register_channel(
        topic="lidar_sim",
        message_encoding=MessageEncoding.JSON,
        schema_id=schema_id
    )
    
    return schema_id, channel_id
