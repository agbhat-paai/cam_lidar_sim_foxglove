import numpy as np
import open3d as o3d
import cv2
import time
import yaml
import json
import base64
from pathlib import Path
from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding
from utils.helpers import setup_simulation, get_pose, setup_foxglove_lidar_schema


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def main():
    cfg = load_config()

    # Simulation Loop Parameters
    fps = cfg['simulation']['fps']
    total_frames = int(cfg['simulation']['duration_sec'] * fps)
    speed_mps = cfg['simulation']['speed_mps']
    debug_viz = cfg['simulation'].get('debug_viz', False)

    lidar_radius = cfg['lidar']['range_m']
    channels = cfg['lidar']['channels']  
    vertical_fov_deg = cfg['lidar']['vertical_fov_deg']
    horizontal_fov_deg = cfg['lidar']['horizontal_fov_deg']
    lidar_accuracy = cfg['lidar']['accuracy_m']
    angle_threshold_rad = np.deg2rad(cfg['lidar'].get('angle_threshold_deg', 0.5))


    datapath, output_mcap, start_pos, R, motion_vec, K = setup_simulation(cfg, sensor_type='lidar')

    print(f"Loading point cloud from {datapath}...")
    pcd = o3d.io.read_point_cloud(datapath)
    
    #downsample if enabled
    if cfg['processing']['downsample_enabled']:
        voxel_size = cfg['processing']['voxel_size_m']
        print(f"Downsampling with voxel size {voxel_size}m...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # offset points to UTM origin as described by the authors
    points = np.asarray(pcd.points) - np.array(cfg['world']['utm_offset'])
    pcd.points = o3d.utility.Vector3dVector(points)

    points_world = np.asarray(pcd.points).T
    colors_world = np.asarray(pcd.colors).T
    print(f"Loaded {points_world.shape[1]} points.")

    kdtree = o3d.geometry.KDTreeFlann(pcd)

    BEV_IMAGE_SIZE = 800  # 800x800 pixel window
    PIXELS_PER_METER = 10 
    
    if debug_viz:
        print("Debug visualization enabled. A window will open.")
        # Create a resizable window
        cv2.namedWindow("LiDAR BEV Simulation", cv2.WINDOW_NORMAL)


    with open(output_mcap, "wb") as f:
        writer = Writer(f)
        writer.start()

        # register schema and channel for foxglove point cloud
        schema_id, channel_id = setup_foxglove_lidar_schema(writer)

        print(f"Starting simulation ({total_frames} frames)...")
        start_ns = time.time_ns()

        for i in range(total_frames):
            start_process = time.time()
            
            # get the current pose
            curr_pose = get_pose(i, total_frames, start_pos, motion_vec, speed_mps, fps)

            # lets make a bubble with KD-Tree for faster point selection for large point clouds
            [k, idx, _] = kdtree.search_radius_vector_3d(curr_pose, lidar_radius)
            if len(idx) == 0: continue

            print("Points reduced to :", len(idx))

            local_points = points_world[:, idx]
            local_colors = colors_world[:, idx]
            # o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local_points.T))])

            #CORE LOGIC
            #convert local points relative to lidar frame
            points_lidar = local_points - curr_pose.reshape(3, 1)

            #calculate spherical coordinates
            x, y, z = points_lidar
            radius = np.linalg.norm(points_lidar, axis=0)
            elevation = np.arcsin(z / radius)  # vertical angle

            # print(radius.shape, elevation.shape, azimuth.shape)

            #based on no. of beams, filter
            elevation_arr = np.linspace(
                np.deg2rad(vertical_fov_deg[0]), 
                np.deg2rad(vertical_fov_deg[1]), 
                channels)
            
            # print(elevation_arr)

            selected_mask = np.zeros(len(idx), dtype=bool)

            for j, point in enumerate(points_lidar.T):
                x,y,z = point
                
                r = np.linalg.norm(point, axis=0) + 1e-9
                safe_ratio = np.clip(z / r, -1.0, 1.0)
                elev = np.arcsin(safe_ratio)
                # azim = np.degrees(np.arctan2(y, x)) 

                nearest_elev_rad = elevation_arr[np.argmin(np.abs(elevation_arr - elev))]

                if abs(nearest_elev_rad - elev) <= angle_threshold_rad:
                    selected_mask[j] = True

            final_points_lidar = points_lidar[:, selected_mask]
            final_colors_lidar = local_colors[:, selected_mask]
            num_points = final_points_lidar.shape[1]


            #thank you gemini
            if debug_viz:
                # 1. Create a blank black image
                bev_image = np.zeros((BEV_IMAGE_SIZE, BEV_IMAGE_SIZE, 3), dtype=np.uint8)

                # 2. Get the (x, y) points from our filtered scan
                x_points = final_points_lidar[0, :]
                y_points = final_points_lidar[1, :]

                # 3. Convert from meters to pixel coordinates
                # Center of image (400, 400) is the car (0, 0)
                # We flip the y-axis (-) so that positive Y (forward) is "up" in the image
                u_pix = (x_points * PIXELS_PER_METER + BEV_IMAGE_SIZE / 2).astype(int)
                v_pix = (-y_points * PIXELS_PER_METER + BEV_IMAGE_SIZE / 2).astype(int)

                # 4. Filter out points that are outside our 800x800 view
                in_bounds = (u_pix >= 0) & (u_pix < BEV_IMAGE_SIZE) & \
                            (v_pix >= 0) & (v_pix < BEV_IMAGE_SIZE)
                
                u_pix_in = u_pix[in_bounds]
                v_pix_in = v_pix[in_bounds]
                colors_in = final_colors_lidar[:, in_bounds]

                # 5. Paint the points onto the image
                if colors_in.shape[1] > 0:
                    # Convert (0-1) RGB to (0-255) BGR for OpenCV
                    colors_bgr = (colors_in.T * 255).astype(np.uint8)[:, [2,1,0]]
                    bev_image[v_pix_in, u_pix_in] = colors_bgr

                # 6. Draw the "car" at the center
                # A 20x10 pixel red rectangle
                car_center = BEV_IMAGE_SIZE // 2
                cv2.rectangle(bev_image, 
                              (car_center - 5, car_center - 10), 
                              (car_center + 5, car_center + 10), 
                              (0, 0, 255), 2) # Red (BGR)

                # 7. Display the image
                cv2.imshow("LiDAR BEV Simulation", bev_image)
                
                # 8. Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nUser pressed 'q', stopping simulation.")
                    debug_viz = False # Stop visualizing
                    break # Exit the main loop
        
            if num_points == 0:
                continue # skip frame if no points are visible


            end_process = time.time()
            print(f"Frame {i+1}/{total_frames} processed in {(end_process - start_process)*1000:.1f} ms ({num_points} points)")
        

    if debug_viz:
        cv2.destroyAllWindows()

    print(f"\nLidar simulation finished! Output: {output_mcap}")
    

                

            


    

if __name__ == "__main__":
    main()