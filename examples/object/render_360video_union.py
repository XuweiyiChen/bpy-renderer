import json
import os
from glob import glob

import imageio
import numpy as np

from bpyrenderer import SceneManager
from bpyrenderer.camera import add_camera
from bpyrenderer.camera.layout import get_camera_positions_on_sphere
from bpyrenderer.engine import init_render_engine
from bpyrenderer.environment import set_background_color, set_env_map
from bpyrenderer.importer import load_armature, load_file
from bpyrenderer.render_output import (
    enable_color_output,
    enable_depth_output,
    enable_normals_output,
)
from bpyrenderer.utils import convert_depth_to_webp, convert_normal_to_webp

output_dir = "outputs"

# 1. Init engine and scene manager
init_render_engine("BLENDER_EEVEE")
scene_manager = SceneManager()
scene_manager.clear(reset_keyframes=True)

# 2. Import models
load_file("../../assets/models/glb_example.glb")

# Others. smooth objects and normalize scene
scene_manager.smooth()
scene_manager.normalize_scene(1.0)
scene_manager.set_material_transparency(False)
scene_manager.set_materials_opaque()  # !!! Important for render normal but may cause render error !!!
scene_manager.normalize_scene(1.0)

# 3. Set environment
set_env_map("../../assets/env_textures/brown_photostudio_02_1k.exr")
# set_background_color([1.0, 1.0, 1.0, 1.0])

# 4. Prepare cameras
cam_pos, cam_mats, elevations, azimuths = get_camera_positions_on_sphere(
    center=(0, 0, 0),
    radius=1.5,
    elevations=[15],
    num_camera_per_layer=120,
    azimuth_offset=-90,  # forward issue
)
cameras = []
for i, camera_mat in enumerate(cam_mats):
    camera = add_camera(camera_mat, add_frame=i < len(cam_mats) - 1)
    cameras.append(camera)

# 5. Set render outputs
width, height, fps = 1024, 1024, 24
enable_color_output(
    width,
    height,
    output_dir,
    mode="PNG",
    film_transparent=True,
)
enable_depth_output(output_dir)
enable_normals_output(output_dir)

scene_manager.render()

# convert depth (.exr) into .png
render_files = sorted(glob(os.path.join(output_dir, "depth_*.exr")))
output_files = [file.replace("exr", "png") for file in render_files]
min_depth, scale = convert_depth_to_webp(render_files, output_files)
for filepath in render_files:
    os.remove(filepath)

# convert normal (.exr) into .png
for file in os.listdir(output_dir):
    if file.startswith("normal_") and file.endswith(".exr"):
        filepath = os.path.join(output_dir, file)
        render_filepath = filepath.replace("normal_", "render_")
        convert_normal_to_webp(
            filepath,
            filepath.replace(".exr", ".png"),
            render_filepath,
        )
        os.remove(filepath)

# convert rendered render_*.png (rgba) to a white background video and a mask video
render_files = sorted(glob(os.path.join(output_dir, "render_*.png")))
if render_files:
    # Create videos for white background and mask
    white_video_path = os.path.join(output_dir, "rgb.mp4")
    mask_video_path = os.path.join(output_dir, "mask.mp4")

    with imageio.get_writer(
        white_video_path, fps=fps
    ) as white_writer, imageio.get_writer(mask_video_path, fps=fps) as mask_writer:

        for file in render_files:
            # Read RGBA image
            image = imageio.imread(file)
            mask = image[:, :, 3]
            white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255

            alpha = image[:, :, 3:4] / 255.0
            white_image = image[:, :, :3] * alpha + white_bg * (1 - alpha)

            white_writer.append_data(white_image.astype(np.uint8))
            mask_writer.append_data(mask)
            os.remove(file)

# convert rendered normal_*.png to a video, and remove original images
normal_files = sorted(glob(os.path.join(output_dir, "normal_*.png")))
if normal_files:
    video_path = os.path.join(output_dir, "normal.mp4")
    with imageio.get_writer(video_path, fps=fps) as writer:
        for file in normal_files:
            image = imageio.imread(file)
            writer.append_data(image)
            os.remove(file)

# convert rendered depth_*.png to a video, and remove original images
depth_files = sorted(glob(os.path.join(output_dir, "depth_*.png")))
if depth_files:
    video_path = os.path.join(output_dir, "depth.mp4")
    with imageio.get_writer(video_path, fps=fps) as writer:
        for file in depth_files:
            image = imageio.imread(file)
            writer.append_data(image)
            os.remove(file)

# save metadata
meta_info = {"width": width, "height": height, "locations": []}
for i in range(len(cam_pos)):
    index = "{0:04d}".format(i)
    meta_info["locations"].append(
        {
            "index": index,
            # intristic
            "projection_type": cameras[i].data.type,
            "ortho_scale": cameras[i].data.ortho_scale,
            "camera_angle_x": cameras[i].data.angle_x,
            # extristic
            "elevation": elevations[i],
            "azimuth": azimuths[i],
            "transform_matrix": cam_mats[i].tolist(),
            # depth
            "depth_min": float(min_depth),
            "depth_scale": float(scale),
        }
    )
with open(os.path.join(output_dir, "meta.json"), "w") as f:
    json.dump(meta_info, f, indent=4)
