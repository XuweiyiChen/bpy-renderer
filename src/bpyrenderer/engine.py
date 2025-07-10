from typing import Literal

import bpy


def init_render_engine(
    engine: Literal["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"], render_samples: int = 64
):
    """Initialize the rendering engine.

    Args:
        engine (Literal[&quot;CYCLES&quot;, &quot;BLENDER_EEVEE&quot;, &quot;BLENDER_EEVEE_NEXT&quot;]):
            The rendering engine to use. Either CYCLES, BLENDER_EEVEE, or BLENDER_EEVEE_NEXT.
        render_samples (int, optional):
            Number of samples to render. Defaults to 64.

    Raises:
        ValueError: If the engine is not CYCLES, BLENDER_EEVEE, or BLENDER_EEVEE_NEXT.
    """
    if engine == "CYCLES":
        cycles_init(render_samples)
    elif engine == "BLENDER_EEVEE":
        eevee_init(render_samples)
    elif engine == "BLENDER_EEVEE_NEXT":
        eevee_next_init(render_samples)
    else:
        raise ValueError(f"Unknown engine: {engine}")


def eevee_init(render_samples: int):
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.eevee.taa_render_samples = render_samples
    bpy.context.scene.eevee.use_gtao = True
    bpy.context.scene.eevee.use_ssr = True
    bpy.context.scene.eevee.use_bloom = True
    bpy.context.scene.render.use_high_quality_normals = True


def eevee_next_init(render_samples: int):
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    # EEVEE Next uses different properties
    bpy.context.scene.eevee.taa_render_samples = render_samples
    bpy.context.scene.render.use_high_quality_normals = True


def cycles_init(render_samples: int):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = render_samples
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3
    bpy.context.scene.cycles.transmission_bounces = 3
    bpy.context.scene.cycles.filter_width = 0.01
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.film_transparent = True
