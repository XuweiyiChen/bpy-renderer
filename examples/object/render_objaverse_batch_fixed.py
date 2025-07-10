#!/usr/bin/env python3
"""
Fixed batch rendering script for Objaverse GLB files using threading.
Renders RGB, depth, and mask outputs for all GLB files in a directory.
"""

import os
import json
import time
import numpy as np
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: OpenCV not available, mask generation will be limited")

from bpyrenderer.camera import add_camera
from bpyrenderer.engine import init_render_engine
from bpyrenderer.environment import set_background_color, set_env_map
from bpyrenderer.importer import load_file
from bpyrenderer.render_output import enable_color_output, enable_depth_output
from bpyrenderer.utils import convert_depth_to_webp
from bpyrenderer import SceneManager
from bpyrenderer.camera.layout import get_camera_positions_on_sphere

# Thread-local storage for scene managers
thread_local = threading.local()

def get_scene_manager():
    """Get or create a scene manager for the current thread"""
    if not hasattr(thread_local, 'scene_manager'):
        # Initialize engine and scene manager for this thread
        init_render_engine("BLENDER_EEVEE_NEXT")
        thread_local.scene_manager = SceneManager()
    return thread_local.scene_manager


def render_glb_multiview(glb_path, output_base_dir, env_texture_path=None, 
                        width=512, height=512, num_views=12):
    """
    Render a single GLB file from multiple viewpoints.
    
    Args:
        glb_path: Path to the GLB file
        output_base_dir: Base output directory
        env_texture_path: Path to environment texture (optional)
        width, height: Image dimensions
        num_views: Number of camera viewpoints
    
    Returns:
        Dictionary with rendering results
    """
    
    try:
        # Get GLB filename without extension
        glb_name = Path(glb_path).stem
        
        # Create output directories for this GLB
        glb_output_dir = os.path.join(output_base_dir, glb_name)
        rgb_dir = os.path.join(glb_output_dir, "rgb")
        depth_dir = os.path.join(glb_output_dir, "depth")
        mask_dir = os.path.join(glb_output_dir, "mask")
        
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        thread_id = threading.get_ident()
        print(f"Thread {thread_id}: Starting render for {glb_name}")
        
        # Get scene manager for this thread
        scene_manager = get_scene_manager()
        scene_manager.clear(reset_keyframes=True)
        
        # Import the GLB model
        load_file(glb_path)
        
        # Smooth objects and normalize scene
        scene_manager.smooth()
        scene_manager.normalize_scene(1.0)
        
        # Set environment
        if env_texture_path and os.path.exists(env_texture_path):
            set_env_map(env_texture_path)
        else:
            # Use a neutral gray background if no environment texture
            set_background_color([0.5, 0.5, 0.5])
        
        # Prepare camera positions
        cam_pos, cam_mats, elevations, azimuths = get_camera_positions_on_sphere(
            center=(0, 0, 0),
            radius=1.5,
            elevations=[15, 45],  # Two elevation levels
            num_camera_per_layer=num_views // 2,
            azimuth_offset=-90,
        )
        
        # Render each view
        frame_info_list = []
        for i, (pos, mat, elev, azim) in enumerate(zip(cam_pos, cam_mats, elevations, azimuths)):
            # Clear previous cameras
            scene_manager.clear(reset_keyframes=False)
            
            # Add camera for this view
            camera = add_camera(mat, add_frame=False)
            
            frame_index = "{:04d}".format(i)
            
            # Enable RGB output with transparent background for mask generation
            enable_color_output(
                width, height, rgb_dir,
                file_prefix=f"frame{frame_index}",
                file_format="PNG",
                mode="IMAGE",
                film_transparent=True,
            )
            
            # Enable depth output
            enable_depth_output(
                output_dir=depth_dir,
                file_prefix=f"frame{frame_index}"
            )
            
            # Render this view
            scene_manager.render()
            
            # Generate mask from transparent PNG
            rgb_path = os.path.join(rgb_dir, f"frame{frame_index}.png")
            mask_path = os.path.join(mask_dir, f"frame{frame_index}.png")
            
            # Create mask using alpha channel
            if os.path.exists(rgb_path) and cv2 is not None:
                img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
                if img is not None and img.shape[2] == 4:  # RGBA
                    # Create binary mask from alpha channel
                    alpha = img[:, :, 3]
                    mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
                    cv2.imwrite(mask_path, mask)
                else:
                    # Fallback: create mask from non-black pixels
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        mask = np.where(gray > 10, 255, 0).astype(np.uint8)
                        cv2.imwrite(mask_path, mask)
            
            # Store frame info
            frame_info = {
                "frame_index": frame_index,
                "rgb_file": f"rgb/frame{frame_index}.png",
                "depth_file": f"depth/frame{frame_index}.png",
                "mask_file": f"mask/frame{frame_index}.png",
                "camera_properties": {
                    "elevation": elev,
                    "azimuth": azim,
                    "transform_matrix": mat.tolist(),
                    "camera_position": list(pos)
                }
            }
            frame_info_list.append(frame_info)
        
        # Convert depth images
        depth_exr_files = sorted(glob(os.path.join(depth_dir, "frame*.exr")))
        depth_png_files = [f.replace('.exr', '.png') for f in depth_exr_files]
        
        min_depth, scale = 0.0, 1.0
        if depth_exr_files:
            min_depth, scale = convert_depth_to_webp(depth_exr_files, depth_png_files)
        
        # Save metadata for this GLB
        metadata = {
            "glb_file": glb_path,
            "glb_name": glb_name,
            "width": width,
            "height": height,
            "num_views": len(frame_info_list),
            "depth_info": {
                "min_depth": min_depth,
                "scale": scale,
                "reconstruction_formula": "world_depth = png_value / scale + min_depth"
            },
            "camera_info": {
                "radius": 1.5,
                "elevations": [15, 45],
                "azimuth_offset": -90
            },
            "frames": frame_info_list
        }
        
        with open(os.path.join(glb_output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Thread {thread_id}: Completed render for {glb_name} - {len(frame_info_list)} views")
        
        return {
            "glb_name": glb_name,
            "status": "success",
            "num_views": len(frame_info_list),
            "output_dir": glb_output_dir
        }
        
    except Exception as e:
        print(f"Thread {threading.get_ident()}: Error rendering {glb_path}: {str(e)}")
        return {
            "glb_name": Path(glb_path).stem if glb_path else "unknown",
            "status": "error",
            "error": str(e),
            "output_dir": None
        }


def render_glb_batch(glb_files, output_base_dir, env_texture_path=None, 
                    width=512, height=512, num_views=12, n_jobs=4):
    """
    Render a batch of GLB files using threading.
    
    Args:
        glb_files: List of GLB file paths
        output_base_dir: Base output directory
        env_texture_path: Path to environment texture (optional)
        width, height: Image dimensions
        num_views: Number of camera viewpoints per GLB
        n_jobs: Number of parallel threads
    
    Returns:
        List of results for each GLB file
    """
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"Starting batch render of {len(glb_files)} GLB files using {n_jobs} parallel threads")
    print(f"Output directory: {output_base_dir}")
    print(f"Image resolution: {width}x{height}")
    print(f"Views per object: {num_views}")
    
    start_time = time.time()
    results = []
    
    # Use ThreadPoolExecutor for parallel rendering
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_glb = {
            executor.submit(
                render_glb_multiview, 
                glb_file, output_base_dir, env_texture_path, width, height, num_views
            ): glb_file for glb_file in glb_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_glb):
            glb_file = future_to_glb[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed: {result['glb_name']} ({len(results)}/{len(glb_files)})")
            except Exception as exc:
                print(f"GLB {glb_file} generated an exception: {exc}")
                results.append({
                    "glb_name": Path(glb_file).stem,
                    "status": "error",
                    "error": str(exc),
                    "output_dir": None
                })
    
    end_time = time.time()
    
    # Summary
    successful = sum(1 for r in results if r and r["status"] == "success") if results else 0
    failed = sum(1 for r in results if r and r["status"] == "error") if results else 0
    total_views = sum(r.get("num_views", 0) for r in results if r and r["status"] == "success") if results else 0
    
    print(f"\n{'='*60}")
    print(f"BATCH RENDERING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Successfully rendered: {successful} GLB files")
    print(f"Failed: {failed} GLB files")
    print(f"Total views rendered: {total_views}")
    print(f"Average time per GLB: {(end_time - start_time) / len(glb_files):.2f} seconds")
    
    # Save batch summary
    summary = {
        "batch_info": {
            "total_glbs": len(glb_files),
            "successful": successful,
            "failed": failed,
            "total_views": total_views,
            "total_time_seconds": end_time - start_time,
            "avg_time_per_glb": (end_time - start_time) / len(glb_files)
        },
        "settings": {
            "width": width,
            "height": height,
            "num_views": num_views,
            "n_jobs": n_jobs,
            "env_texture_path": env_texture_path
        },
        "results": results
    }
    
    with open(os.path.join(output_base_dir, "batch_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    return results


def main():
    """Main function to run the batch rendering"""
    
    # Configuration
    GLB_DIR = "/home/ubuntu/xuweiyi/bpy-renderer/objaverse/glbs/000-023"
    OUTPUT_DIR = "/home/ubuntu/xuweiyi/bpy-renderer/objaverse_renders"
    ENV_TEXTURE = "/home/ubuntu/xuweiyi/bpy-renderer/assets/env_textures/brown_photostudio_02_1k.exr"
    
    # Rendering settings
    WIDTH = 512
    HEIGHT = 512
    NUM_VIEWS = 12  # 6 views at each of 2 elevation levels
    N_JOBS = 4  # Reduced from 8 to 4 for threading (still efficient)
    
    # Find all GLB files
    glb_pattern = os.path.join(GLB_DIR, "*.glb")
    glb_files = sorted(glob(glb_pattern))
    
    if not glb_files:
        print(f"No GLB files found in {GLB_DIR}")
        return
    
    print(f"Found {len(glb_files)} GLB files to render")
    
    # Check if environment texture exists
    if not os.path.exists(ENV_TEXTURE):
        print(f"Warning: Environment texture not found at {ENV_TEXTURE}")
        ENV_TEXTURE = None
    
    # Run batch rendering
    results = render_glb_batch(
        glb_files=glb_files,
        output_base_dir=OUTPUT_DIR,
        env_texture_path=ENV_TEXTURE,
        width=WIDTH,
        height=HEIGHT,
        num_views=NUM_VIEWS,
        n_jobs=N_JOBS
    )
    
    print(f"\nBatch rendering completed!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Check batch_summary.json for detailed results")


if __name__ == "__main__":
    main() 