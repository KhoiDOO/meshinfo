import ctypes

import trimesh

import numpy as np
from OpenGL.GL import *

from ..constants import COLOR_OFFSET, VERTEX_STRIDE

from ..analysis.mesh import MeshInfo


class MeshBuffer:
    def __init__(self):
        self.mesh: trimesh.Trimesh = None
        self.mesh_info: MeshInfo = None
        self.intersected_face_ids = None
        self.normal_length = 0.0
        self.points = None
        self.point_normals = None
        self.bounds = None
        self.bounds_center = None
        self.bounds_size = None
        self.original_bounds = None
        self.original_bounds_center = None
        self.original_bounds_size = None
        self.position = np.zeros(3, dtype=np.float32)

        # Main Buffer
        self.main_vao = glGenVertexArrays(1)
        self.main_vbo = glGenBuffers(1)
        self.main_ebo = glGenBuffers(1)
        self.main_index_count = 0

        # Intersected Faces Buffer
        self.intersected_vao = glGenVertexArrays(1)
        self.intersected_vbo = glGenBuffers(1)
        self.intersected_ebo = glGenBuffers(1)
        self.intersected_index_count = 0

        # Face Normals Buffer (lines)
        self.face_normals_vao = glGenVertexArrays(1)
        self.face_normals_vbo = glGenBuffers(1)
        self.face_normals_count = 0

        # Vertex Normals Buffer (lines)
        self.vertex_normals_vao = glGenVertexArrays(1)
        self.vertex_normals_vbo = glGenBuffers(1)
        self.vertex_normals_count = 0

        # Point Cloud Buffer (points)
        self.point_cloud_vao = glGenVertexArrays(1)
        self.point_cloud_vbo = glGenBuffers(1)
        self.point_cloud_count = 0

        # Point Cloud Normals Buffer (lines)
        self.point_cloud_normals_vao = glGenVertexArrays(1)
        self.point_cloud_normals_vbo = glGenBuffers(1)
        self.point_cloud_normals_count = 0

        # Non-manifold Edges Buffer (lines)
        self.nonmanifold_edges_vao = glGenVertexArrays(1)
        self.nonmanifold_edges_vbo = glGenBuffers(1)
        self.nonmanifold_edges_count = 0

        # Non-manifold Vertices Buffer (points)
        self.nonmanifold_vertices_vao = glGenVertexArrays(1)
        self.nonmanifold_vertices_vbo = glGenBuffers(1)
        self.nonmanifold_vertices_count = 0

    def update_from_mesh(
        self, 
        mesh: trimesh.Trimesh, 
        mesh_info: MeshInfo, 
        normal_length: float, 
        points: np.ndarray, 
        point_normals: np.ndarray, 
        color_scheme: dict
    ):
        self.mesh = mesh
        self.mesh_info = mesh_info
        self.intersected_face_ids = mesh_info.intersected_face_ids
        self.normal_length = normal_length
        self.points = points
        self.point_normals = point_normals
        self.bounds = mesh.bounds
        self.bounds_center = (self.bounds[0] + self.bounds[1]) * 0.5
        self.bounds_size = self.bounds[1] - self.bounds[0]
        # Store original unscaled bounds for layout calculations
        self.original_bounds = np.copy(self.bounds)
        self.original_bounds_center = np.copy(self.bounds_center)
        self.original_bounds_size = np.copy(self.bounds_size)
        self.update_gpu_buffers(color_scheme)

    def refresh_colors(self, color_scheme: dict):
        if self.mesh is None:
            return
        self.update_gpu_buffers(color_scheme)

    def update_gpu_buffers(self, color_scheme: dict):
        # Split faces into two groups
        all_indices = np.arange(len(self.mesh.faces))
        intersected_mask = np.array([i in self.intersected_face_ids for i in all_indices])

        # 1. Prepare Main Mesh (Not highlighted)
        # main_faces = self.mesh.faces[~intersected_mask]
        main_faces = self.mesh.faces
        self.main_index_count = self.setup_buffer(self.main_vao, self.main_vbo, self.main_ebo, main_faces, color_scheme)

        # 2. Prepare Intersected Faces (Selected)
        intersected_faces = self.mesh.faces[intersected_mask]
        self.intersected_index_count = self.setup_buffer(
            self.intersected_vao,
            self.intersected_vbo,
            self.intersected_ebo,
            intersected_faces,
            color_scheme,
        )

        # 3. Prepare Normals
        self.face_normals_count = self.setup_face_normals_buffer(color_scheme)
        self.vertex_normals_count = self.setup_vertex_normals_buffer(color_scheme)

        # 4. Prepare Point Cloud
        self.point_cloud_count, self.point_cloud_normals_count = self.setup_point_cloud_buffer(color_scheme)

        # 5. Prepare Non-manifold Edges
        self.nonmanifold_edges_count = self.setup_nonmanifold_edges_buffer(color_scheme)

        # 6. Prepare Non-manifold Vertices
        self.nonmanifold_vertices_count = self.setup_nonmanifold_vertices_buffer(color_scheme)

    def setup_buffer(self, vao, vbo, ebo, faces, color_scheme: dict):
        if len(faces) == 0:
            return 0

        # Un-index vertices so each triangle has unique data (easier for coloring/normals)
        vertices = self.mesh.vertices[faces].reshape(-1, 3)
        colors = np.full((vertices.shape[0], 3), color_scheme["mesh"], dtype=np.float32)

        data = np.hstack((vertices, colors)).astype(np.float32)
        indices = np.arange(len(vertices)).astype(np.uint32)

        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_DYNAMIC_DRAW)

        # Layout: Pos(3), Color(3)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(COLOR_OFFSET))
        glEnableVertexAttribArray(1)

        return len(indices)

    def setup_point_cloud_buffer(self, color_scheme: dict):
        points = self.points
        colors = np.full((points.shape[0], 3), color_scheme["point_cloud"], dtype=np.float32)
        data = np.hstack((points, colors)).astype(np.float32)

        glBindVertexArray(self.point_cloud_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.point_cloud_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(COLOR_OFFSET))
        glEnableVertexAttribArray(1)

        # Prepare Point Cloud Normals
        normals = self.point_normals
        line_verts = np.empty((points.shape[0] * 2, 3), dtype=np.float32)
        line_verts[0::2] = points
        line_verts[1::2] = points + normals * self.normal_length

        colors = np.full((line_verts.shape[0], 3), color_scheme["point_cloud_normals"], dtype=np.float32)
        data = np.hstack((line_verts, colors)).astype(np.float32)

        glBindVertexArray(self.point_cloud_normals_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.point_cloud_normals_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(COLOR_OFFSET))
        glEnableVertexAttribArray(1)

        return points.shape[0], line_verts.shape[0]

    def setup_face_normals_buffer(self, color_scheme: dict):
        # Face centers and normals
        centers = self.mesh.triangles_center
        normals = self.mesh.face_normals

        line_verts = np.empty((centers.shape[0] * 2, 3), dtype=np.float32)
        line_verts[0::2] = centers
        line_verts[1::2] = centers + normals * self.normal_length

        colors = np.full((line_verts.shape[0], 3), color_scheme["face_normals"], dtype=np.float32)
        data = np.hstack((line_verts, colors)).astype(np.float32)

        glBindVertexArray(self.face_normals_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.face_normals_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(COLOR_OFFSET))
        glEnableVertexAttribArray(1)

        return line_verts.shape[0]

    def setup_vertex_normals_buffer(self, color_scheme: dict):
        verts = self.mesh.vertices
        normals = self.mesh.vertex_normals

        line_verts = np.empty((verts.shape[0] * 2, 3), dtype=np.float32)
        line_verts[0::2] = verts
        line_verts[1::2] = verts + normals * self.normal_length

        colors = np.full((line_verts.shape[0], 3), color_scheme["vertex_normals"], dtype=np.float32)
        data = np.hstack((line_verts, colors)).astype(np.float32)

        glBindVertexArray(self.vertex_normals_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_normals_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(COLOR_OFFSET))
        glEnableVertexAttribArray(1)

        return line_verts.shape[0]

    def setup_nonmanifold_edges_buffer(self, color_scheme: dict):
        if len(self.mesh_info.nonmanifold_edges) == 0: return 0

        # Get vertex positions for each non-manifold edge
        nonmanifold_edges = self.mesh_info.nonmanifold_edges
        verts = self.mesh.vertices

        # Create line vertices (start and end point for each edge)
        line_verts = np.empty((nonmanifold_edges.shape[0] * 2, 3), dtype=np.float32)
        line_verts[0::2] = verts[nonmanifold_edges[:, 0]]
        line_verts[1::2] = verts[nonmanifold_edges[:, 1]]

        colors = np.full((line_verts.shape[0], 3), color_scheme["nonmanifold_edges"], dtype=np.float32)
        data = np.hstack((line_verts, colors)).astype(np.float32)

        glBindVertexArray(self.nonmanifold_edges_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.nonmanifold_edges_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(COLOR_OFFSET))
        glEnableVertexAttribArray(1)

        return line_verts.shape[0]

    def setup_nonmanifold_vertices_buffer(self, color_scheme: dict):
        if len(self.mesh_info.nonmanifold_vertices) == 0: return 0

        # Get vertex positions for non-manifold vertices
        nonmanifold_vertices = self.mesh_info.nonmanifold_vertices
        verts = self.mesh.vertices

        # Get positions of non-manifold vertices
        vertex_positions = verts[nonmanifold_vertices]

        colors = np.full((vertex_positions.shape[0], 3), color_scheme["nonmanifold_vertices"], dtype=np.float32)
        data = np.hstack((vertex_positions, colors)).astype(np.float32)

        glBindVertexArray(self.nonmanifold_vertices_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.nonmanifold_vertices_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(COLOR_OFFSET))
        glEnableVertexAttribArray(1)

        return vertex_positions.shape[0]
