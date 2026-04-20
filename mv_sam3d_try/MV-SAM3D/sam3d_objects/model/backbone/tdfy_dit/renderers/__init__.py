# Copyright (c) Meta Platforms, Inc. and affiliates.

try:
    from .octree_renderer import OctreeRenderer
except ImportError:
    OctreeRenderer = None

try:
    from .gaussian_render import GaussianRenderer
except ImportError:
    GaussianRenderer = None

# handle case when nvdiffrast is not present on the machine
try:
    from .mesh_renderer import MeshRenderer
except ImportError:
    MeshRenderer = None