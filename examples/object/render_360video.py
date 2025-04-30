import os
import json
import numpy as np

from bpyrenderer.camera import add_camera
from bpyrenderer.engine import init_render_engine
from bpyrenderer.environment import set_background_color, set_env_map
from bpyrenderer.importer import load_file, load_armature
from bpyrenderer.render_output import enable_color_output
from bpyrenderer import SceneManager
from bpyrenderer.camera.layout import get_camera_positions_on_sphere

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
width, height = 1024, 1024
enable_color_output(
    width,
    height,
    output_dir,
    mode="VIDEO",
    film_transparent=False,  # enable rendering result to include background
)
scene_manager.render()

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
