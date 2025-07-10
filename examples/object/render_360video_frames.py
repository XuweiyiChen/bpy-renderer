import os
import json
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm

from bpyrenderer.camera import add_camera
from bpyrenderer.engine import init_render_engine
from bpyrenderer.environment import set_background_color, set_env_map
from bpyrenderer.importer import load_file, load_armature
from bpyrenderer.render_output import enable_color_output, enable_depth_output
from bpyrenderer.utils import convert_depth_to_webp
from bpyrenderer import SceneManager
from bpyrenderer.camera.layout import get_camera_positions_on_sphere

# Configuration
GLB_DIR = "/home/ubuntu/xuweiyi/bpy-renderer/objaverse/glbs/000-023"
OUTPUT_BASE_DIR = "/home/ubuntu/xuweiyi/bpy-renderer/objaverse_renders"
ENV_TEXTURE = "../../assets/env_textures/brown_photostudio_02_1k.exr"

# Find all GLB files
glb_files = list(Path(GLB_DIR).glob("*.glb"))
print(f"Found {len(glb_files)} GLB files to process")

if not glb_files:
    print("No GLB files found!")
    exit()

# Initialize engine and scene manager once
init_render_engine("BLENDER_EEVEE_NEXT")
scene_manager = SceneManager()

# Process each GLB file
for glb_file in tqdm(glb_files, desc="Processing GLB files"):
    glb_name = glb_file.stem
    print(f"\nProcessing: {glb_name}")
    
    # Create output directories for this GLB
    output_dir = os.path.join(OUTPUT_BASE_DIR, glb_name)
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    try:
        # Clear scene and import the GLB model
        scene_manager.clear(reset_keyframes=True)
        load_file(str(glb_file))
        
        # Smooth objects and normalize scene
        scene_manager.smooth()
        scene_manager.normalize_scene(1.0)
        
        # Set environment
        if os.path.exists(ENV_TEXTURE):
            set_env_map(ENV_TEXTURE)
        else:
            set_background_color([0.5, 0.5, 0.5, 1.0])
        
        # Prepare cameras
        cam_pos, cam_mats, elevations, azimuths = get_camera_positions_on_sphere(
            center=(0, 0, 0),
            radius=1.5,
            elevations=[15],
            num_camera_per_layer=12,
            azimuth_offset=-90,
        )
        cameras = []
        for i, camera_mat in enumerate(cam_mats):
            camera = add_camera(camera_mat, add_frame=i < len(cam_mats) - 1)
            cameras.append(camera)
        
        # Set render outputs for individual frames
        width, height = 512, 512
        
        # Enable color output with transparent background
        enable_color_output(
            width,
            height,
            rgb_dir,
            file_prefix="frame",
            file_format="PNG",
            mode="IMAGE",
            film_transparent=True,
        )
        
        # Enable depth output
        enable_depth_output(
            output_dir=depth_dir,
            file_prefix="frame"
        )
        
        # Render all frames
        print(f"Rendering {len(cam_pos)} views...")
        scene_manager.render()
        print("Rendering complete!")
        
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
        
        # Save complete metadata with camera poses and depth info
        meta_info = {
            "glb_file": str(glb_file),
            "glb_name": glb_name,
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
            "frames": []
        }
        
        for i in range(len(cam_pos)):
            frame_index = "{:04d}".format(i)
            frame_info = {
                "frame_index": frame_index,
                "rgb_file": f"rgb/frame{frame_index}.png",
                "depth_file": f"depth/frame{frame_index}.png",
                "camera_properties": {
                    "projection_type": cameras[i].data.type,
                    "ortho_scale": cameras[i].data.ortho_scale,
                    "camera_angle_x": cameras[i].data.angle_x,
                    "elevation": elevations[i],
                    "azimuth": azimuths[i],
                    "transform_matrix": cam_mats[i].tolist(),
                    "camera_position": list(cam_pos[i])
                }
            }
            meta_info["frames"].append(frame_info)
        
        # Save metadata
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(meta_info, f, indent=2)
        
        print(f"✓ Successfully processed {glb_name}")
        print(f"  Output: {output_dir}")
        
    except Exception as e:
        print(f"✗ Error processing {glb_name}: {str(e)}")
        continue

print(f"\nBatch processing complete!")
print(f"Results saved to: {OUTPUT_BASE_DIR}") 