import json
import numpy as np
import os
from PIL import Image
import argparse

def load_metadata(metadata_path):
    """Load camera poses and depth conversion parameters from metadata.json"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def depth_png_to_world(depth_png, min_depth, scale):
    """Convert depth PNG values back to world coordinates"""
    # Formula: world_depth = png_value / scale + min_depth
    world_depth = depth_png.astype(np.float32) / scale + min_depth
    return world_depth

def camera_intrinsics_from_blender(width, height, camera_angle_x):
    """Get camera intrinsics from Blender camera parameters"""
    # Calculate focal length from field of view
    # Blender's camera_angle_x is the horizontal field of view
    focal_length_x = 0.5 * width / np.tan(0.5 * camera_angle_x)
    
    # For square pixels, focal length is the same in both directions
    focal_length_y = focal_length_x
    
    # Principal point (center of image)
    cx = width / 2.0
    cy = height / 2.0
    
    # Intrinsic matrix (standard pinhole camera model)
    K = np.array([
        [focal_length_x, 0, cx],
        [0, focal_length_y, cy],
        [0, 0, 1]
    ])
    
    return K

def unproject_depth_to_3d(depth_image, rgb_image, K, transform_matrix, min_depth, max_depth_threshold=10.0):
    """Unproject depth image to 3D points in world coordinates, filtering out background"""
    height, width = depth_image.shape
    
    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Create mask for object pixels (not background)
    # 1. Valid depth (> min_depth)
    # 2. Not too far (< max_depth_threshold from min_depth)
    # 3. Has alpha channel (if RGBA) or non-zero RGB
    valid_depth_mask = (depth_image > min_depth) & (depth_image < (min_depth + max_depth_threshold))
    
    # Check if RGB image has alpha channel
    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 4:
        # Use alpha channel to mask out background
        alpha_mask = rgb_image[:, :, 3] > 0
        valid_mask = valid_depth_mask & alpha_mask
    else:
        # Use RGB values - background should be black or very dark
        if len(rgb_image.shape) == 3:
            rgb_sum = np.sum(rgb_image, axis=2)
        else:
            rgb_sum = rgb_image
        rgb_mask = rgb_sum > 10  # Threshold for non-background pixels
        valid_mask = valid_depth_mask & rgb_mask
    
    if not np.any(valid_mask):
        return np.empty((0, 3)), valid_mask
    
    # Get valid coordinates
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = depth_image[valid_mask]
    
    # Convert pixel coordinates to camera coordinates
    # Standard computer vision: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = Z
    x_cam = (u_valid - K[0, 2]) * z_valid / K[0, 0]
    y_cam = (v_valid - K[1, 2]) * z_valid / K[1, 1]
    z_cam = z_valid
    
    # Create points in camera coordinate system
    # Blender camera coordinate system: X=right, Y=up, Z=back (towards camera)
    # We need to convert to standard computer vision: X=right, Y=down, Z=forward
    # Blender depth is positive going away from camera (into scene)
    points_cam = np.column_stack([x_cam, -y_cam, -z_cam])  # Flip Y and Z for coordinate system
    
    # Transform matrix from Blender is camera-to-world (camera pose)
    transform_matrix = np.array(transform_matrix)
    
    # Convert to homogeneous coordinates
    points_cam_homo = np.column_stack([points_cam, np.ones(len(points_cam))])
    
    # Transform to world coordinates
    points_world = (transform_matrix @ points_cam_homo.T).T
    
    return points_world[:, :3], valid_mask  # Return mask for color extraction

def reconstruct_point_cloud(data_dir, output_path="pointcloud.ply", max_points_per_frame=50000, max_depth_from_min=5.0):
    """Reconstruct 3D point cloud from RGB, depth, and camera data"""
    
    # Load metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    metadata = load_metadata(metadata_path)
    
    # Extract parameters
    width = metadata["width"]
    height = metadata["height"]
    min_depth = metadata["depth_info"]["min_depth"]
    scale = metadata["depth_info"]["scale"]
    
    print(f"Reconstructing from {metadata['num_frames']} frames")
    print(f"Image size: {width}x{height}")
    print(f"Depth range: min={min_depth:.3f}, scale={scale:.3f}")
    
    all_points = []
    all_colors = []
    
    for frame_info in metadata["frames"]:
        frame_idx = frame_info["frame_index"]
        
        # Load RGB image
        rgb_path = os.path.join(data_dir, frame_info["rgb_file"])
        rgb_image = np.array(Image.open(rgb_path))
        
        # Load depth image
        depth_path = os.path.join(data_dir, frame_info["depth_file"])
        depth_png = np.array(Image.open(depth_path))
        
        # Convert depth to world coordinates
        depth_world = depth_png_to_world(depth_png, min_depth, scale)
        
        # Get camera parameters
        camera_props = frame_info["camera_properties"]
        camera_angle_x = camera_props["camera_angle_x"]
        transform_matrix = camera_props["transform_matrix"]
        
        # Calculate camera intrinsics
        K = camera_intrinsics_from_blender(width, height, camera_angle_x)
        
        # Debug: Print camera info for first frame
        if frame_idx == "0000":
            print(f"Camera angle_x: {camera_angle_x:.4f} radians ({np.degrees(camera_angle_x):.1f} degrees)")
            print(f"Camera intrinsics K:\n{K}")
            print(f"Camera transform matrix:\n{np.array(transform_matrix)}")
            print(f"Camera position: {camera_props['camera_position']}")
        
        # Unproject to 3D with proper masking
        points_3d, valid_mask = unproject_depth_to_3d(depth_world, rgb_image, K, transform_matrix, min_depth, max_depth_from_min)
        
        if len(points_3d) == 0:
            print(f"No valid points in frame {frame_idx}")
            continue
        
        # Get corresponding colors using the same mask
        colors = rgb_image[valid_mask]
        
        # Handle RGBA images - keep only RGB channels
        if len(colors.shape) == 2 and colors.shape[1] == 4:
            colors = colors[:, :3]
        
        # Subsample if too many points
        if len(points_3d) > max_points_per_frame:
            indices = np.random.choice(len(points_3d), max_points_per_frame, replace=False)
            points_3d = points_3d[indices]
            colors = colors[indices]
        
        all_points.append(points_3d)
        all_colors.append(colors)
        
        print(f"Frame {frame_idx}: {len(points_3d)} points")
    
    # Combine all points
    if all_points:
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        print(f"Total points: {len(combined_points)}")
        
        # Print point cloud statistics
        print(f"Point cloud bounds:")
        print(f"  X: [{combined_points[:, 0].min():.3f}, {combined_points[:, 0].max():.3f}]")
        print(f"  Y: [{combined_points[:, 1].min():.3f}, {combined_points[:, 1].max():.3f}]")
        print(f"  Z: [{combined_points[:, 2].min():.3f}, {combined_points[:, 2].max():.3f}]")
        print(f"Point cloud center: [{combined_points[:, 0].mean():.3f}, {combined_points[:, 1].mean():.3f}, {combined_points[:, 2].mean():.3f}]")
        
        # Save as PLY file
        save_ply(combined_points, combined_colors, output_path)
        print(f"Saved point cloud to {output_path}")
    else:
        print("No valid points found!")

def save_ply(points, colors, filename):
    """Save point cloud as PLY file"""
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    with open(filename, 'w') as f:
        f.write(header)
        for point, color in zip(points, colors):
            # Handle both RGB and RGBA
            if len(color) >= 3:
                r, g, b = color[:3]
            else:
                r = g = b = color[0] if len(color) == 1 else 128
            
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(r)} {int(g)} {int(b)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct 3D point cloud from rendered data")
    parser.add_argument("--data_dir", default="outputs_v3", help="Directory containing RGB, depth, and metadata")
    parser.add_argument("--output", default="pointcloud.ply", help="Output PLY file")
    parser.add_argument("--max_points", type=int, default=50000, help="Maximum points per frame")
    parser.add_argument("--max_depth_range", type=float, default=5.0, help="Maximum depth range from min_depth to include")
    
    args = parser.parse_args()
    
    reconstruct_point_cloud(args.data_dir, args.output, args.max_points, args.max_depth_range) 