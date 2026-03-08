import os
import numpy as np

import trimesh
import json

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

def load_pc_xyz(file_path: str):
    
    assert file_path.endswith('.xyz'), "Only .xyz files are supported."
    assert os.path.isfile(file_path), f"File not found: {file_path}"
    
    points = []
    normals = []

    with open(file_path, 'r') as f:
        for line in f:

            if line.startswith('#'):  # Skip comment lines
                continue

            line = line.strip()
            if line:  # Skip empty lines
                splits = line.split()
                assert len(splits) == 3 or len(splits) == 6, "Each line must contain either 3 values (x, y, z) or 6 values (x, y, z, nx, ny, nz)."

                x, y, z = map(float, splits[:3])
                points.append((x, y, z))

                if len(splits) == 6:
                    nx, ny, nz = map(float, splits[3:])
                    normals.append((nx, ny, nz))
                
    points = np.array(points, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32) if normals else None

    return points, normals

def load_pc(file_path: str):
    ext = os.path.basename(file_path).split('.')[-1].lower()
    if ext == 'xyz':
        return load_pc_xyz(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only .xyz files are supported.")