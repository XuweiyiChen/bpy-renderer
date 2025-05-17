# bpy-renderer

A go-to library for rendering 3D scenes and animations. Whether you're looking for a simple rendering script for making demos or producing multi-view image dataset for training, bpy-renderer is a modular toolbox that supports both.

Bpy-renderer offers two core components:

* Core package for setting engines, cameras, environments, models, scenes, rendering outputs.
* [Example scripts](./examples/) for various functions.

## Demos

**3D Object**

https://github.com/user-attachments/assets/6e5a5767-0323-40aa-95a7-f1ab465976d6

**3D Scene**

https://github.com/user-attachments/assets/d68cf607-3769-487d-9cc7-9c8da311abb6

**3D Animation**

https://github.com/user-attachments/assets/2c7e356c-be30-4d73-bce8-2d9a9965a482

https://github.com/user-attachments/assets/bda030ee-9144-4cb4-bab0-0fb2082180b8

## Installation

We recommend installing bpy-renderer in **Python-3.10**.

```Bash
git clone https://github.com/huanngzh/bpy-renderer.git
pip install -e ./bpy-renderer
```

## Quick Start

Coming soon! For now, please check our example script in [render_360video.py](./examples/object/render_360video.py), which renders 360 degree video of a 3D model.

## Example Scripts

| Scripts | Task |
| - | - |
| [render_6ortho.py](examples/object/render_6ortho.py) | Render 6 ortho views with rgb, depth, normals |
| [render_360video.py](examples/object/render_360video.py) | Render 360 degree video |
