import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def load_ply(filepath):
    """Load PLY file and return points and colors"""
    points = []
    colors = []
    
    with open(filepath, 'r') as f:
        # Read header
        line = f.readline()
        while line and not line.startswith('end_header'):
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            line = f.readline()
        
        # Read vertices
        for i in range(num_vertices):
            line = f.readline().strip().split()
            x, y, z = float(line[0]), float(line[1]), float(line[2])
            r, g, b = int(line[3]), int(line[4]), int(line[5])
            
            points.append([x, y, z])
            colors.append([r, g, b])
    
    return np.array(points), np.array(colors)

def analyze_point_cloud(points, colors):
    """Analyze point cloud statistics"""
    print(f"Point cloud analysis:")
    print(f"  Total points: {len(points)}")
    print(f"  Bounds:")
    print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] (range: {points[:, 0].max() - points[:, 0].min():.3f})")
    print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] (range: {points[:, 1].max() - points[:, 1].min():.3f})")
    print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] (range: {points[:, 2].max() - points[:, 2].min():.3f})")
    print(f"  Center: [{points[:, 0].mean():.3f}, {points[:, 1].mean():.3f}, {points[:, 2].mean():.3f}]")
    print(f"  Standard deviation: [{points[:, 0].std():.3f}, {points[:, 1].std():.3f}, {points[:, 2].std():.3f}]")
    
    # Check for outliers
    for axis, axis_name in enumerate(['X', 'Y', 'Z']):
        mean = points[:, axis].mean()
        std = points[:, axis].std()
        outliers = np.abs(points[:, axis] - mean) > 3 * std
        print(f"  {axis_name} outliers (>3σ): {outliers.sum()}")

def visualize_point_cloud(points, colors, subsample=10000):
    """Create a 3D visualization of the point cloud"""
    if len(points) > subsample:
        indices = np.random.choice(len(points), subsample, replace=False)
        points_vis = points[indices]
        colors_vis = colors[indices] / 255.0
    else:
        points_vis = points
        colors_vis = colors / 255.0
    
    fig = plt.figure(figsize=(12, 4))
    
    # 3D plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
               c=colors_vis, s=1, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Point Cloud')
    ax1.set_box_aspect([1,1,1])
    
    # XY projection
    ax2 = fig.add_subplot(132)
    ax2.scatter(points_vis[:, 0], points_vis[:, 1], c=colors_vis, s=1, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection (Top View)')
    ax2.set_aspect('equal')
    
    # XZ projection  
    ax3 = fig.add_subplot(133)
    ax3.scatter(points_vis[:, 0], points_vis[:, 2], c=colors_vis, s=1, alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection (Front View)')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('pointcloud_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to pointcloud_analysis.png")

def create_density_map(points, resolution=64):
    """Create a 3D density map of the point cloud"""
    # Create voxel grid
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    # Create regular grid
    ranges = max_coords - min_coords
    voxel_size = ranges.max() / resolution
    
    # Discretize points
    voxel_coords = ((points - min_coords) / voxel_size).astype(int)
    voxel_coords = np.clip(voxel_coords, 0, resolution - 1)
    
    # Count points in each voxel
    density = np.zeros((resolution, resolution, resolution))
    for coord in voxel_coords:
        density[coord[0], coord[1], coord[2]] += 1
    
    occupied_voxels = (density > 0).sum()
    total_voxels = resolution ** 3
    
    print(f"Voxel analysis (resolution {resolution}³):")
    print(f"  Occupied voxels: {occupied_voxels} / {total_voxels} ({100*occupied_voxels/total_voxels:.1f}%)")
    print(f"  Average points per occupied voxel: {points.shape[0] / occupied_voxels:.1f}")
    
    return density

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze reconstructed point cloud")
    parser.add_argument("--input", default="pointcloud_filtered.ply", help="Input PLY file")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--voxel_resolution", type=int, default=64, help="Voxel grid resolution for density analysis")
    
    args = parser.parse_args()
    
    print(f"Loading point cloud from {args.input}")
    points, colors = load_ply(args.input)
    
    analyze_point_cloud(points, colors)
    
    if args.visualize:
        visualize_point_cloud(points, colors)
    
    create_density_map(points, args.voxel_resolution) 