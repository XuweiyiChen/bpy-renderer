import os
import json
import numpy as np

from bpyrenderer.camera import add_camera
from bpyrenderer.engine import init_render_engine
from bpyrenderer.environment import set_background_color, set_env_map
from bpyrenderer.importer import load_file, load_armature
from bpyrenderer.render_output import (
    enable_color_output,
    enable_albedo_output,
    enable_depth_output,
    enable_normals_output,
)
from bpyrenderer import SceneManager
from bpyrenderer.camera.layout import get_camera_positions_on_sphere
from bpyrenderer.utils import convert_normal_to_webp

output_dir = "outputs"

# 1. Init engine and scene manager
init_render_engine("BLENDER_EEVEE")
scene_manager = SceneManager()
scene_manager.clear(reset_keyframes=True)

# 2. Import models
load_file("../../assets/models/glb_example.glb")

# Others. smooth objects and normalize scene
scene_manager.smooth()
scene_manager.clear_normal_map()
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
    elevations=[0],
    azimuths=[item - 90 for item in [0, 45, 90, 180, 270, 315]],  # forward issue
)
cameras = []
for i, camera_mat in enumerate(cam_mats):
    camera = add_camera(camera_mat, "ORTHO", add_frame=i < len(cam_mats) - 1)
    cameras.append(camera)

# 5. Set render outputs
width, height = 1024, 1024
enable_color_output(
    width,
    height,
    output_dir,
    file_format="WEBP",
    mode="IMAGE",
    film_transparent=True,
)
enable_depth_output(output_dir)
enable_normals_output(output_dir)
scene_manager.render()

# Optional. convert normal (.exr) into .webp
for file in os.listdir(output_dir):
    if file.startswith("normal_") and file.endswith(".exr"):
        filepath = os.path.join(output_dir, file)
        render_filepath = filepath.replace("normal_", "render_").replace(
            ".exr", ".webp"
        )
        convert_normal_to_webp(
            filepath,
            filepath.replace(".exr", ".webp"),
            render_filepath,
        )
        os.remove(filepath)

# Optional. save metadata
meta_info = {"width": width, "height": height, "locations": []}
for i in range(len(cam_pos)):
    index = "{0:04d}".format(i)
    meta_info["locations"].append(
        {
            "index": index,
            "projection_type": cameras[i].data.type,
            "ortho_scale": cameras[i].data.ortho_scale,
            "camera_angle_x": cameras[i].data.angle_x,
            "elevation": elevations[i],
            "azimuth": azimuths[i],
            "transform_matrix": cam_mats[i].tolist(),
        }
    )
with open(os.path.join(output_dir, "meta.json"), "w") as f:
    json.dump(meta_info, f, indent=4)
