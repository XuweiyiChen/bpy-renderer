import os
import json
import numpy as np
from glob import glob

from bpyrenderer.camera import add_camera
from bpyrenderer.engine import init_render_engine
from bpyrenderer.environment import set_background_color, set_env_map
from bpyrenderer.importer import load_file, load_armature
from bpyrenderer.render_output import enable_color_output, enable_depth_output
from bpyrenderer.utils import convert_depth_to_webp
from bpyrenderer import SceneManager
from bpyrenderer.camera.layout import get_camera_positions_on_sphere

# Create output directories
output_dir = "outputs_v3"
rgb_dir = os.path.join(output_dir, "rgb")
depth_dir = os.path.join(output_dir, "depth")
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# 1. Init engine and scene manager
init_render_engine("BLENDER_EEVEE_NEXT")
scene_manager = SceneManager()
scene_manager.clear(reset_keyframes=True)

# 2. Import models
load_file("../../assets/models/glb_example.glb")

# Others. smooth objects and normalize scene
scene_manager.smooth()
scene_manager.normalize_scene(1.0)

# 3. Set environment with transparent background
set_env_map("../../assets/env_textures/brown_photostudio_02_1k.exr")
# For transparent background, don't set background color

# 4. Prepare cameras
cam_pos, cam_mats, elevations, azimuths = get_camera_positions_on_sphere(
    center=(0, 0, 0),
    radius=1.5,
    elevations=[15],
    num_camera_per_layer=12,
    azimuth_offset=-90,  # forward issue
)
cameras = []
for i, camera_mat in enumerate(cam_mats):
    camera = add_camera(camera_mat, add_frame=i < len(cam_mats) - 1)
    cameras.append(camera)

# 5. Set render outputs for individual frames
width, height = 1024, 1024

# Enable color output with transparent background
enable_color_output(
    width,
    height,
    rgb_dir,
    file_prefix="frame",
    file_format="PNG",
    mode="IMAGE",
    film_transparent=True,  # Enable transparent background
)

# Enable depth output
enable_depth_output(
    output_dir=depth_dir,
    file_prefix="frame"
)

# 6. Render all frames
print("Starting render...")
scene_manager.render()
print("Rendering complete!")

# 7. Convert depth images from EXR to PNG
print("Converting depth images...")
depth_exr_files = sorted(glob(os.path.join(depth_dir, "frame*.exr")))
depth_png_files = [f.replace('.exr', '.png') for f in depth_exr_files]

if depth_exr_files:
    min_depth, scale = convert_depth_to_webp(depth_exr_files, depth_png_files)
    print(f"Depth conversion complete. Min depth: {min_depth:.4f}, Scale: {scale:.4f}")
else:
    min_depth, scale = 0.0, 1.0
    print("No depth files found to convert")

# 8. Save complete metadata with camera poses and depth info
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

print(f"Saved {len(cam_pos)} frames with RGB, depth, and camera poses to {output_dir}")
print(f"RGB images: {rgb_dir}")
print(f"Depth images: {depth_dir}")
print(f"Metadata: {os.path.join(output_dir, 'metadata.json')}") 