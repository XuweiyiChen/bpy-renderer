import bpy
import math
import numpy as np

import bpy
import mathutils
from mathutils import Vector
from typing import Optional, Literal
from .utils import get_keyframes


class SceneManager:
    @property
    def objects(self):
        return bpy.context.scene.objects

    @property
    def scene_meshes(self):
        return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    @property
    def scene_armatures(self):
        return [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]

    @property
    def data_meshes(self):
        return [obj for obj in bpy.data.objects if obj.type == "MESH"]

    @property
    def root_objects(self):
        for obj in bpy.context.scene.objects.values():
            if not obj.parent:
                yield obj

    @property
    def num_frames(self):
        return bpy.context.scene.frame_end + 1

    def get_scene_bbox(self, single_obj=None, ignore_matrix=False):
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3

        meshes = self.scene_meshes if single_obj is None else [single_obj]
        if len(meshes) == 0:
            raise RuntimeError("No objects in scene to compute bounding box for")

        for obj in meshes:
            for coord in obj.bound_box:
                coord = Vector(coord)
                if not ignore_matrix:
                    coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

        return Vector(bbox_min), Vector(bbox_max)

    def get_scene_bbox_all_frames(self):
        """Get bounding box that contains the entire animation sequence"""
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3

        # Store current frame
        current_frame = bpy.context.scene.frame_current

        # Iterate through all frames
        for frame in range(
            bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1
        ):
            bpy.context.scene.frame_set(frame)
            frame_min, frame_max = self.get_scene_bbox()
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, frame_min))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, frame_max))

        # Restore original frame
        bpy.context.scene.frame_set(current_frame)

        return Vector(bbox_min), Vector(bbox_max)

    def normalize_scene(
        self,
        normalize_range: float = 1.0,
        range_type: Literal["CUBE", "SPHERE"] = "CUBE",
        process_frames: bool = False,
        use_parent_node: bool = False,
    ):
        # Recompute bounding box, offset and scale
        if process_frames:
            bbox_min, bbox_max = self.get_scene_bbox_all_frames()
        else:
            bbox_min, bbox_max = self.get_scene_bbox()

        if range_type == "CUBE":
            scale = normalize_range / max(bbox_max - bbox_min)
        elif range_type == "SPHERE":
            scale = normalize_range / (bbox_max - bbox_min).length
        else:
            raise ValueError(
                f"Invalid range_type: {range_type}. Must be either 'CUBE' or 'SPHERE'"
            )

        # Calculate offset to center
        offset = -(bbox_min + bbox_max) / 2

        if use_parent_node:
            # Create a new empty object as parent
            parent = bpy.data.objects.new("NormalizationNode", None)
            bpy.context.scene.collection.objects.link(parent)

            # Parent all root objects to the new node
            for obj in self.root_objects:
                if obj is not parent:
                    obj.parent = parent
                    # Keep the object's local transform
                    obj.matrix_parent_inverse = parent.matrix_world.inverted()

            # Set parent's location and scale
            parent.scale = (scale, scale, scale)
            parent.location = offset * scale  # !!!important for use_parent_node!!!
        else:
            # Original behavior: modify each object directly
            for obj in self.root_objects:
                obj.matrix_world.translation += offset
                # Scale relative to world center by adjusting translation and scale
                original_translation = obj.matrix_world.translation.copy()
                obj.matrix_world.translation = original_translation * scale
                obj.scale = obj.scale * scale
                bpy.context.view_layer.update()

        # Restore original frame
        bpy.ops.object.select_all(action="DESELECT")

    def rotate_model(self, object, rotateQuaternion):
        object.select_set(True)
        bpy.context.view_layer.objects.active = object
        object.rotation_mode = "QUATERNION"
        object.rotation_quaternion = mathutils.Quaternion(rotateQuaternion)
        bpy.ops.object.transform_apply()

    def render(self):
        bpy.context.scene.render.use_compositing = True
        bpy.context.scene.use_nodes = True

        tree = bpy.context.scene.node_tree
        if "Render Layers" not in tree.nodes:
            tree.nodes.new("CompositorNodeRLayers")
        else:
            tree.nodes["Render Layers"]

        bpy.ops.render.render(animation=True, write_still=True)

    def smooth(self):
        for obj in self.scene_meshes:
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = np.deg2rad(30)

    def clear_normal_map(self):
        for material in bpy.data.materials:
            material.use_nodes = True
            node_tree = material.node_tree
            try:
                bsdf = node_tree.nodes["Principled BSDF"]
                if bsdf.inputs["Normal"].is_linked:
                    for link in bsdf.inputs["Normal"].links:
                        node_tree.links.remove(link)
            except:
                pass

    def set_material_transparency(self, show_transparent_back: bool) -> None:
        """Set transparency settings for materials with blend mode 'BLEND'.

        Args:
            show_transparent_back: Whether to show the back face of transparent materials.
        """
        for material in bpy.data.materials:
            if not material.use_nodes:
                continue

            if material.blend_method == "BLEND":
                material.show_transparent_back = show_transparent_back

    def set_materials_opaque(self) -> None:
        """Set all materials to opaque blend mode.

        This is useful for rendering passes like normal maps that require
        fully opaque materials for correct results.
        """
        for material in bpy.data.materials:
            if not material.use_nodes:
                continue

            material.blend_method = "OPAQUE"

    def update_scene_frames(
        self, mode: Literal["auto", "manual"] = "auto", num_frames: Optional[int] = None
    ):
        if mode == "auto":
            armatures = self.scene_armatures
            keyframes = get_keyframes(armatures)
            bpy.context.scene.frame_end = (
                int(max(keyframes)) if len(keyframes) > 0 else 0
            )
        elif mode == "manual":
            if num_frames is None:
                raise ValueError(f"num_frames must be provided if the mode is 'manual'")
            bpy.context.scene.frame_end = num_frames - 1

    def clear(
        self,
        clear_objects: Optional[bool] = True,
        clear_nodes: Optional[bool] = True,
        reset_keyframes: Optional[bool] = True,
    ):
        if clear_objects:
            objects = [x for x in bpy.data.objects]
            for obj in objects:
                bpy.data.objects.remove(obj, do_unlink=True)

        # Clear all nodes
        if clear_nodes:
            bpy.context.scene.use_nodes = True
            node_tree = bpy.context.scene.node_tree
            for node in node_tree.nodes:
                node_tree.nodes.remove(node)

        # Reset keyframes
        if reset_keyframes:
            bpy.context.scene.frame_start = 0
            bpy.context.scene.frame_end = 0
            for a in bpy.data.actions:
                bpy.data.actions.remove(a)

    def gc(self):
        for _ in range(10):
            bpy.ops.outliner.orphans_purge()
