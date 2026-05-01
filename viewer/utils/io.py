import numpy as np

import trimesh

from ..constants import NORMALIZE_BOUND

def normalize_vertices(vertices: np.ndarray, bound=NORMALIZE_BOUND) -> np.ndarray:
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    ori_center = (vmax + vmin) / 2
    ori_scale = 2 * bound / np.max(vmax - vmin)
    vertices = (vertices - ori_center) * ori_scale
    return vertices

def load_mesh(file_path):
    mesh: trimesh.Trimesh = trimesh.load(file_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = mesh.vertices
    faces = mesh.faces

    # vertices = normalize_vertices(vertices)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    return mesh