import meshlib.mrmeshpy as mrmeshpy
import meshlib.mrmeshnumpy as mrmeshnumpy
import multiprocessing
from functools import partial

import trimesh
import numpy as np

def get_intersected_tria_ids(mesh: trimesh.Trimesh) -> tuple[list[int], int]:
    # 1. Build the MeshLib Mesh
    # MeshLib expects float32 for vertices and int32 for faces
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)
    mrmesh_obj = mrmeshnumpy.meshFromFacesVerts(faces, vertices)

    # 2. Find self-intersections
    # findSelfCollidingTriangles returns a list of FaceFace pairs
    # It internally ignores adjacent triangles (sharing vertices)
    pairs = mrmeshpy.findSelfCollidingTriangles(mrmesh_obj)

    intersected_ids = set()
    
    # 3. Process all results
    for pair in pairs:
        intersected_ids.add(pair.aFace.get())
        intersected_ids.add(pair.bFace.get())

    return list(intersected_ids), len(pairs)

def _is_butterfly_vertex(vertex_idx, face_indices, mesh_faces, already_nonmanifold):
    if vertex_idx in already_nonmanifold or len(face_indices) < 2:
        return None
    
    num_local_faces = len(face_indices)
    local_faces = mesh_faces[face_indices]
    
    # Adjacency list for faces sharing this vertex
    adj = [[] for _ in range(num_local_faces)]
    for i in range(num_local_faces):
        f1 = local_faces[i]
        for j in range(i + 1, num_local_faces):
            f2 = local_faces[j]
            # Two faces are adjacent if they share an edge containing vertex_idx.
            # Since they already share vertex_idx, they share an edge if they share one more vertex.
            # Small set intersection is fast.
            if len(set(f1) & set(f2)) >= 2:
                adj[i].append(j)
                adj[j].append(i)
    
    # BFS to check if all faces form a single connected component
    visited = [False] * num_local_faces
    queue = [0]
    visited[0] = True
    count = 1
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
                
    if count < num_local_faces:
        return vertex_idx
    return None

def get_nonmanifold_vertices(mesh: trimesh.Trimesh, edges_unique: np.ndarray, edges_counts: np.ndarray) -> np.ndarray:
    """
    Find vertices that are non-manifold.
    A vertex is non-manifold if:
    1. It is part of a non-manifold edge (shared by > 2 faces).
    2. Its adjacent faces do not form a single connected component (butterfly vertex).
    """
    nonmanifold_vertices = set()
    
    # 1. Vertices on non-manifold edges (count > 2)
    nonmanifold_edge_mask = edges_counts > 2
    if np.any(nonmanifold_edge_mask):
        nonmanifold_edge_vertices = edges_unique[nonmanifold_edge_mask].flatten()
        nonmanifold_vertices.update(nonmanifold_edge_vertices)
    
    # 2. Check for "butterfly" vertices (connected components of adjacent faces)
    # trimesh.vertex_faces is a padded 2D array, where -1 indicates no face
    vertex_faces = mesh.vertex_faces
    mesh_faces = mesh.faces
    
    # Prepare arguments for parallel processing
    tasks = []
    for vertex_idx, face_indices in enumerate(vertex_faces):
        # Filter out the padding (-1)
        valid_face_indices = face_indices[face_indices != -1]
        if vertex_idx not in nonmanifold_vertices and len(valid_face_indices) >= 2:
            tasks.append((vertex_idx, valid_face_indices))
    if tasks:
        if len(tasks) < 10000:
            # Sequential for small number of tasks to avoid multiprocessing overhead
            results = [_is_butterfly_vertex(v_idx, f_indices, mesh_faces, nonmanifold_vertices) for v_idx, f_indices in tasks]
        else:
            # Use multiprocessing to check vertices in parallel for large meshes
            num_cores = multiprocessing.cpu_count()
            chunksize = max(1, len(tasks) // (num_cores * 4))

            check_func = partial(
                _is_butterfly_vertex, 
                mesh_faces=mesh_faces, 
                already_nonmanifold=nonmanifold_vertices
            )

            with multiprocessing.Pool(processes=num_cores) as pool:
                results = pool.starmap(check_func, tasks, chunksize=chunksize)

        for v in results:
            if v is not None:
                nonmanifold_vertices.add(v)

    
    return np.array(sorted(list(nonmanifold_vertices)), dtype=np.int32)

def get_num_dup_faces(mesh: trimesh.Trimesh) -> int:
    # Count duplicate faces by sorting the vertex indices of each face
    sorted_faces = np.sort(mesh.faces, axis=1)
    unique_faces, counts = np.unique(sorted_faces, axis=0, return_counts=True)
    num_dup_faces = np.sum(counts[counts > 1] - 1).item()  # Count duplicates beyond the first occurrence
    return num_dup_faces

def get_sphericity(volume, area) -> float:
    if volume == 0 or area == 0:
        return 0.0
    sphericity = (np.pi ** (1/3)) * ((6 * abs(volume)) ** (2/3)) / area
    return sphericity