import math
import os
import json
from typing import List, Optional, Literal, Tuple, Dict

import bpy
import mathutils
import numpy as np
from PIL import Image, ImageDraw
import bpy_extras

from .utils import get_local2world_mat, PRESET_COLORS


def enable_color_output(
    width: int,
    height: int,
    output_dir: Optional[str] = "",
    file_prefix: str = "render_",
    file_format: Literal["WEBP", "PNG"] = "WEBP",
    mode: Literal["IMAGE", "VIDEO"] = "IMAGE",
    **kwargs,
):
    film_transparent = kwargs.get("film_transparent", True)
    fps = kwargs.get("fps", 24)

    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = film_transparent
    scene.render.image_settings.quality = 100

    if mode == "IMAGE":
        scene.render.image_settings.file_format = file_format
        scene.render.image_settings.color_mode = "RGBA"
        # scene.render.image_settings.color_depth = "16"
    elif mode == "VIDEO":
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.fps = fps
    scene.render.filepath = os.path.join(output_dir, file_prefix)


def enable_normals_output(output_dir: Optional[str] = "", file_prefix: str = "normal_"):
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_normal = True
    bpy.context.scene.render.use_compositing = True

    separate_rgba = tree.nodes.new("CompositorNodeSepRGBA")
    space_between_nodes_x = 200
    space_between_nodes_y = -300
    separate_rgba.location.x = space_between_nodes_x
    separate_rgba.location.y = space_between_nodes_y
    tree.links.new(rl.outputs["Normal"], separate_rgba.inputs["Image"])

    combine_rgba = tree.nodes.new("CompositorNodeCombRGBA")
    combine_rgba.location.x = space_between_nodes_x * 14

    c_channels = ["R", "G", "B"]
    offset = space_between_nodes_x * 2
    multiplication_values: List[List[bpy.types.Node]] = [[], [], []]
    channel_results = {}
    for row_index, channel in enumerate(c_channels):
        # matrix multiplication
        mulitpliers = []
        for column in range(3):
            multiply = tree.nodes.new("CompositorNodeMath")
            multiply.operation = "MULTIPLY"
            # setting at the end for all frames
            multiply.inputs[1].default_value = 0
            multiply.location.x = column * space_between_nodes_x + offset
            multiply.location.y = row_index * space_between_nodes_y
            tree.links.new(
                separate_rgba.outputs[c_channels[column]], multiply.inputs[0]
            )
            mulitpliers.append(multiply)
            multiplication_values[row_index].append(multiply)

        first_add = tree.nodes.new("CompositorNodeMath")
        first_add.operation = "ADD"
        first_add.location.x = space_between_nodes_x * 5 + offset
        first_add.location.y = row_index * space_between_nodes_y
        tree.links.new(mulitpliers[0].outputs["Value"], first_add.inputs[0])
        tree.links.new(mulitpliers[1].outputs["Value"], first_add.inputs[1])

        second_add = tree.nodes.new("CompositorNodeMath")
        second_add.operation = "ADD"
        second_add.location.x = space_between_nodes_x * 6 + offset
        second_add.location.y = row_index * space_between_nodes_y
        tree.links.new(first_add.outputs["Value"], second_add.inputs[0])
        tree.links.new(mulitpliers[2].outputs["Value"], second_add.inputs[1])

        channel_results[channel] = second_add

    rot_around_x_axis = mathutils.Matrix.Rotation(math.radians(-90.0), 4, "X")
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
        bpy.context.scene.frame_set(frame)
        used_rotation_matrix = (
            get_local2world_mat(bpy.context.scene.camera) @ rot_around_x_axis
        )
        for row_index in range(3):
            for column_index in range(3):
                current_multiply = multiplication_values[row_index][column_index]
                current_multiply.inputs[1].default_value = used_rotation_matrix[
                    column_index
                ][row_index]
                current_multiply.inputs[1].keyframe_insert(
                    data_path="default_value", frame=frame
                )

    offset = 8 * space_between_nodes_x
    for index, channel in enumerate(c_channels):
        multiply = tree.nodes.new("CompositorNodeMath")
        multiply.operation = "MULTIPLY"
        multiply.location.x = space_between_nodes_x * 2 + offset
        multiply.location.y = index * space_between_nodes_y
        tree.links.new(channel_results[channel].outputs["Value"], multiply.inputs[0])
        multiply.inputs[1].default_value = 0.5
        if channel == "G":
            multiply.inputs[1].default_value = -0.5
        add = tree.nodes.new("CompositorNodeMath")
        add.operation = "ADD"
        add.location.x = space_between_nodes_x * 3 + offset
        add.location.y = index * space_between_nodes_y
        tree.links.new(multiply.outputs["Value"], add.inputs[0])
        add.inputs[1].default_value = 0.5
        output_channel = channel
        if channel == "G":
            output_channel = "B"
        elif channel == "B":
            output_channel = "G"
        tree.links.new(add.outputs["Value"], combine_rgba.inputs[output_channel])

    normal_file_output = tree.nodes.new("CompositorNodeOutputFile")
    normal_file_output.base_path = output_dir
    normal_file_output.format.file_format = "OPEN_EXR"
    normal_file_output.format.color_mode = "RGBA"
    normal_file_output.format.color_depth = "32"
    normal_file_output.location.x = space_between_nodes_x * 15
    normal_file_output.file_slots.values()[0].path = file_prefix
    tree.links.new(combine_rgba.outputs["Image"], normal_file_output.inputs["Image"])


def enable_depth_output(output_dir: Optional[str] = "", file_prefix: str = "depth_"):
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    links = tree.links

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_z = True

    depth_output = tree.nodes.new("CompositorNodeOutputFile")
    depth_output.base_path = output_dir
    depth_output.name = "DepthOutput"
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_depth = "32"
    depth_output.file_slots.values()[0].path = file_prefix

    links.new(rl.outputs["Depth"], depth_output.inputs["Image"])


def enable_albedo_output(output_dir: Optional[str] = "", file_prefix: str = "albedo_"):
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_diffuse_color = True

    alpha_albedo = tree.nodes.new(type="CompositorNodeSetAlpha")
    tree.links.new(rl.outputs["DiffCol"], alpha_albedo.inputs["Image"])
    tree.links.new(rl.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

    albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.base_path = output_dir
    albedo_file_output.file_slots[0].use_node_format = True
    albedo_file_output.format.file_format = "PNG"
    albedo_file_output.format.color_mode = "RGBA"
    albedo_file_output.format.color_depth = "16"
    albedo_file_output.file_slots.values()[0].path = file_prefix

    tree.links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])


def enable_pbr_output(output_dir, attr_name, color_mode="RGBA", file_prefix: str = ""):
    if file_prefix == "":
        file_prefix = attr_name.lower().replace(" ", "-") + "_"

    for material in bpy.data.materials:
        material.use_nodes = True
        node_tree = material.node_tree
        nodes = node_tree.nodes
        roughness_input = nodes["Principled BSDF"].inputs[attr_name]
        if roughness_input.is_linked:
            linked_node = roughness_input.links[0].from_node
            linked_socket = roughness_input.links[0].from_socket

            aov_output = nodes.new("ShaderNodeOutputAOV")
            aov_output.name = attr_name
            node_tree.links.new(linked_socket, aov_output.inputs[0])

        else:
            fixed_roughness = roughness_input.default_value
            if isinstance(fixed_roughness, float):
                roughness_value = nodes.new("ShaderNodeValue")
                input_idx = 1
            else:
                roughness_value = nodes.new("ShaderNodeRGB")
                input_idx = 0

            roughness_value.outputs[0].default_value = fixed_roughness

            aov_output = nodes.new("ShaderNodeOutputAOV")
            aov_output.name = attr_name
            node_tree.links.new(roughness_value.outputs[0], aov_output.inputs[0])

    tree = bpy.context.scene.node_tree
    links = tree.links
    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]

    roughness_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    roughness_file_output.base_path = output_dir
    roughness_file_output.file_slots[0].use_node_format = True
    roughness_file_output.format.file_format = "PNG"
    roughness_file_output.format.color_mode = color_mode
    roughness_file_output.format.color_depth = "16"
    roughness_file_output.file_slots.values()[0].path = file_prefix

    bpy.ops.scene.view_layer_add_aov()
    bpy.context.scene.view_layers["ViewLayer"].active_aov.name = attr_name
    roughness_alpha = tree.nodes.new(type="CompositorNodeSetAlpha")
    tree.links.new(rl.outputs[attr_name], roughness_alpha.inputs["Image"])
    tree.links.new(rl.outputs["Alpha"], roughness_alpha.inputs["Alpha"])

    links.new(roughness_alpha.outputs["Image"], roughness_file_output.inputs["Image"])


def get_keypoint_data(keypoint_names: Optional[List] = None):
    # Set keypoint colors
    keypoint_colors = {}
    if keypoint_names is None:
        for obj in bpy.context.scene.objects:
            if obj.type == "ARMATURE":
                for i, bone in enumerate(obj.pose.bones):
                    keypoint_colors[bone.name.lower().split(":")[-1]] = PRESET_COLORS[
                        i % len(PRESET_COLORS)
                    ]
    elif isinstance(keypoint_names, dict):
        keypoint_colors = keypoint_names
    elif isinstance(keypoint_names, list):
        keypoint_colors = {
            keypoint_names[i]: PRESET_COLORS[i] for i in range(len(keypoint_names))
        }
    else:
        raise ValueError("keypoint_names must be a list or dictionary")

    # Get camera and scene dimensions
    camera = bpy.context.scene.camera
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y

    # Store keypoint data for all bones
    keypoint_data = {}

    # Process each armature in the scene
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            for bone in obj.pose.bones:
                if not bone.name.lower().split(":")[-1] in keypoint_colors.keys():
                    continue

                # Get bone head in world coordinates
                head_world = obj.matrix_world @ bone.head

                # Project to 2D using Blender's projection
                head_2d = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, camera, head_world
                )

                # Convert to screen coordinates
                head_screen = (int(head_2d.x * width), int((1 - head_2d.y) * height))

                # get color
                keypoint_color = keypoint_colors.get(
                    bone.name.lower().split(":")[-1], list(keypoint_colors.values())[0]
                )

                # Get parent bone name if exists
                parent_name = bone.parent.name if bone.parent else None

                # Store data for this bone
                keypoint_data[bone.name] = {
                    "head_3d": list(head_world),
                    "head_2d": head_screen,
                    "color": keypoint_color,
                    "parent": parent_name,
                }

    return keypoint_data


def visualize_keypoint_map(
    keypoint_data: Dict,
    bone_color: Tuple = (200, 200, 200),
    bone_width: int = 3,
    keypoint_radius: int = 3,
    color_mode: Literal["RGBA", "RGB"] = "RGB",
    plot_bones: Optional[List] = None,
):
    """Visualize the skeleton of armature in the scene using only head positions
    and parent-child relationships. Creates a 2D visualization using projected
    coordinates from camera view.
    """
    # Get camera and scene dimensions
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y

    # Create a new image with transparent background
    if color_mode == "RGB":
        image = Image.new("RGB", (width, height), (0, 0, 0))
    elif color_mode == "RGBA":
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    else:
        raise ValueError(
            f"Unsupported color mode: {color_mode}. Must be either 'RGB' or 'RGBA'."
        )
    draw = ImageDraw.Draw(image)

    # Draw keypoints and connections
    for bone_name, data in list(keypoint_data.items())[::-1]:
        # Check if this bone should be plotted
        plot_bone = (
            bone_name.lower().split(":")[-1] in plot_bones
            if plot_bones is not None
            else True
        )

        if not plot_bone:
            continue

        # Draw connection to parent if exists
        if data["parent"] and data["parent"] in keypoint_data:
            parent_data = keypoint_data[data["parent"]]
            draw.line(
                [data["head_2d"], parent_data["head_2d"]],
                fill=bone_color,
                width=bone_width,
            )

        # Draw keypoint
        draw.ellipse(
            [
                data["head_2d"][0] - keypoint_radius,
                data["head_2d"][1] - keypoint_radius,
                data["head_2d"][0] + keypoint_radius,
                data["head_2d"][1] + keypoint_radius,
            ],
            fill=data["color"],
        )

    return image


def render_keypoint_map(
    output_dir: Optional[str] = "",
    file_prefix: str = "keypoint_",
    file_format: Literal["WEBP", "PNG"] = "WEBP",
    export_meta: bool = True,
    **kwargs,
):
    # kwargs for visualizing the bones
    keypoint_names = kwargs.get("keypoint_names", None)
    bone_color = kwargs.get("bone_color", (200, 200, 200))
    bone_width = kwargs.get("bone_width", 3)
    keypoint_radius = kwargs.get("keypoint_radius", 3)
    plot_bones = kwargs.get("plot_bones", None)

    # Save the image
    os.makedirs(output_dir, exist_ok=True)
    file_suffix = ".png" if file_format == "PNG" else ".webp"

    # Initialize metadata
    keypoint_metas = {
        "bone_names": [],
        "head_3d": [],
        "head_2d": [],
        "colors": [],
        "parents": [],
    }

    original_frame = bpy.context.scene.frame_current
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        output_path = os.path.join(output_dir, f"{file_prefix}{frame:04d}{file_suffix}")
        keypoint_data = get_keypoint_data(keypoint_names)
        keypoint_map = visualize_keypoint_map(
            keypoint_data=keypoint_data,
            bone_color=bone_color,
            bone_width=bone_width,
            keypoint_radius=keypoint_radius,
            plot_bones=plot_bones,
        )
        keypoint_map.save(output_path, file_format, quality=100)

        if export_meta:
            # Initialize frame data
            if frame == bpy.context.scene.frame_start:
                keypoint_metas["bone_names"] = list(keypoint_data.keys())
                keypoint_metas["colors"] = np.array(
                    [data["color"] for data in keypoint_data.values()]
                )
                keypoint_metas["parents"] = [
                    data["parent"] for data in keypoint_data.values()
                ]

            # Collect frame data
            frame_head_3d = np.array(
                [data["head_3d"] for data in keypoint_data.values()]
            )
            frame_head_2d = np.array(
                [data["head_2d"] for data in keypoint_data.values()]
            )

            keypoint_metas["head_3d"].append(frame_head_3d)
            keypoint_metas["head_2d"].append(frame_head_2d)

    if export_meta:
        # Convert lists to numpy arrays
        keypoint_metas["head_3d"] = np.array(keypoint_metas["head_3d"])
        keypoint_metas["head_2d"] = np.array(keypoint_metas["head_2d"])

        # Save metadata
        meta_file = os.path.join(output_dir, f"{file_prefix.split('_')[0]}.npy")
        np.save(meta_file, keypoint_metas)

    bpy.context.scene.frame_set(original_frame)
