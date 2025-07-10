import os
import json
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import multiprocessing

from bpyrenderer.camera import add_camera
from bpyrenderer.engine import init_render_engine
from bpyrenderer.environment import set_background_color, set_env_map
from bpyrenderer.importer import load_file, load_armature
from bpyrenderer.render_output import enable_color_output, enable_depth_output
from bpyrenderer.utils import convert_depth_to_webp
from bpyrenderer import SceneManager
from bpyrenderer.camera.layout import get_camera_positions_on_sphere


def setup_scene(model_path, env_texture_path):
    """Setup the Blender scene - this needs to be done in each process"""
    # 1. Init engine and scene manager
    init_render_engine("BLENDER_EEVEE_NEXT")
    scene_manager = SceneManager()
    scene_manager.clear(reset_keyframes=True)

    # 2. Import models
    load_file(model_path)

    # Others. smooth objects and normalize scene
    scene_manager.smooth()
    scene_manager.normalize_scene(1.0)

    # 3. Set environment with transparent background
    set_env_map(env_texture_path)
    
    return scene_manager


def render_camera_chunk(camera_positions, camera_matrices, elevations, azimuths, 
                       chunk_indices, output_dir, width, height, model_path, env_texture_path):
    """
    Render a chunk of camera positions in a separate process
    
    Args:
        camera_positions: List of camera positions
        camera_matrices: List of camera transformation matrices
        elevations: List of elevation angles
        azimuths: List of azimuth angles
        chunk_indices: Indices of cameras to render in this chunk
        output_dir: Output directory
        width, height: Image dimensions
        model_path: Path to 3D model
        env_texture_path: Path to environment texture
    
    Returns:
        List of dictionaries with frame information
    """
    
    # Setup scene in this process
    scene_manager = setup_scene(model_path, env_texture_path)
    
    # Create output directories for this chunk
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    chunk_results = []
    
    for i in chunk_indices:
        # Clear previous cameras
        scene_manager.clear(reset_keyframes=False)
        
        # Add camera for this frame
        camera = add_camera(camera_matrices[i], add_frame=False)
        
        # Set frame-specific output paths
        frame_index = "{:04d}".format(i)
        
        # Enable outputs for this specific frame
        enable_color_output(
            width,
            height,
            rgb_dir,
            file_prefix=f"frame{frame_index}",
            file_format="PNG",
            mode="IMAGE",
            film_transparent=True,
        )
        
        enable_depth_output(
            output_dir=depth_dir,
            file_prefix=f"frame{frame_index}"
        )
        
        # Render this frame
        print(f"Process {os.getpid()}: Rendering frame {frame_index}")
        scene_manager.render()
        
        # Store frame info
        frame_info = {
            "frame_index": frame_index,
            "rgb_file": f"rgb/frame{frame_index}.png",
            "depth_file": f"depth/frame{frame_index}.png",
            "camera_properties": {
                "projection_type": camera.data.type,
                "ortho_scale": camera.data.ortho_scale,
                "camera_angle_x": camera.data.angle_x,
                "elevation": elevations[i],
                "azimuth": azimuths[i],
                "transform_matrix": camera_matrices[i].tolist(),
                "camera_position": list(camera_positions[i])
            }
        }
        chunk_results.append(frame_info)
    
    return chunk_results


def render_360video_parallel(model_path="../../assets/models/glb_example.glb",
                            env_texture_path="../../assets/env_textures/brown_photostudio_02_1k.exr",
                            output_dir="outputs_v3_parallel",
                            n_jobs=None):
    """
    Parallel version of 360 video rendering
    
    Args:
        model_path: Path to 3D model
        env_texture_path: Path to environment texture
        output_dir: Output directory
        n_jobs: Number of parallel jobs (None = auto-detect)
    """
    
    # Create output directories
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    # Prepare cameras (same as original)
    cam_pos, cam_mats, elevations, azimuths = get_camera_positions_on_sphere(
        center=(0, 0, 0),
        radius=1.5,
        elevations=[15],
        num_camera_per_layer=12,
        azimuth_offset=-90,
    )
    
    width, height = 1024, 1024
    
    # Determine number of jobs
    if n_jobs is None:
        n_jobs = min(multiprocessing.cpu_count(), len(cam_pos))
    
    print(f"Using {n_jobs} parallel jobs for {len(cam_pos)} frames")
    
    # Split camera indices into chunks
    camera_indices = list(range(len(cam_pos)))
    chunk_size = max(1, len(camera_indices) // n_jobs)
    chunks = [camera_indices[i:i + chunk_size] for i in range(0, len(camera_indices), chunk_size)]
    
    # Parallel rendering
    print("Starting parallel rendering...")
    all_frame_results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(render_camera_chunk)(
            cam_pos, cam_mats, elevations, azimuths,
            chunk, output_dir, width, height, 
            model_path, env_texture_path
        ) for chunk in chunks
    )
    
    # Flatten results and sort by frame index
    all_frames = []
    if all_frame_results:
        for chunk_results in all_frame_results:
            if chunk_results:
                all_frames.extend(chunk_results)
    
    # Sort frames by frame index
    all_frames.sort(key=lambda x: x['frame_index'])
    
    print("Parallel rendering complete!")
    
    # Convert depth images from EXR to PNG
    print("Converting depth images...")
    depth_exr_files = sorted(glob(os.path.join(depth_dir, "frame*.exr")))
    depth_png_files = [f.replace('.exr', '.png') for f in depth_exr_files]
    
    if depth_exr_files:
        min_depth, scale = convert_depth_to_webp(depth_exr_files, depth_png_files)
        print(f"Depth conversion complete. Min depth: {min_depth:.4f}, Scale: {scale:.4f}")
    else:
        min_depth, scale = 0.0, 1.0
        print("No depth files found to convert")
    
    # Save complete metadata
    meta_info = {
        "width": width, 
        "height": height, 
        "num_frames": len(cam_pos),
        "depth_info": {
            "min_depth": min_depth,
            "scale": scale,
            "reconstruction_formula": "world_depth = png_value / scale + min_depth"
        },
        "camera_info": {
            "radius": 1.5,
            "elevations": elevations,
            "azimuth_offset": -90
        },
        "frames": all_frames
    }
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta_info, f, indent=2)
    
    print(f"Saved {len(cam_pos)} frames with RGB, depth, and camera poses to {output_dir}")
    print(f"RGB images: {rgb_dir}")
    print(f"Depth images: {depth_dir}")
    print(f"Metadata: {os.path.join(output_dir, 'metadata.json')}")


if __name__ == "__main__":
    # Run with different parallelization levels
    
    # Option 1: Auto-detect number of cores
    render_360video_parallel()
    
    # Option 2: Specify number of jobs
    # render_360video_parallel(n_jobs=4)
    
    # Option 3: Single-threaded (equivalent to original)
    # render_360video_parallel(n_jobs=1) 