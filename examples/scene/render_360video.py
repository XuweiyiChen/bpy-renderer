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
from bpyrenderer.render_output import enable_color_output

output_dir = "outputs"
model_path = "../../assets/models/compo_scene_created_by_midi3d.glb"
env_map = "../../assets/env_textures/brown_photostudio_02_1k.exr"

# 1. Init engine and scene manager
init_render_engine("BLENDER_EEVEE")
scene_manager = SceneManager()
scene_manager.clear(reset_keyframes=True)

# 2. Import models
load_file(model_path)

# Others. smooth objects and normalize scene
scene_manager.smooth()
scene_manager.normalize_scene(1.0)

# 3. Set environment
set_env_map(env_map)
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
    mode="PNG",
    film_transparent=True,  # enable rendering result to include background
)
scene_manager.render()


# convert rendered render_*.png (rgba) to a white background video and a mask video
render_files = sorted(glob(os.path.join(output_dir, "render_*.png")))
if render_files:
    # Create videos for white background and mask
    white_video_path = os.path.join(output_dir, "rgb.mp4")
    mask_video_path = os.path.join(output_dir, "mask.mp4")
    fps = 24

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
