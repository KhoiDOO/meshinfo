import moderngl
import trimesh
import numpy as np

from ..constants import VERTEX_STRIDE
from ..analysis.mesh import MeshInfo

class MeshBuffer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
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
        self.main_vao = None
        self.main_vbo = None
        self.main_ebo = None
        self.main_index_count = 0

        # Intersected Faces Buffer
        self.intersected_vao = None
        self.intersected_vbo = None
        self.intersected_ebo = None
        self.intersected_index_count = 0

        # Face Normals Buffer (lines)
        self.face_normals_vao = None
        self.face_normals_vbo = None
        self.face_normals_count = 0

        # Vertex Normals Buffer (lines)
        self.vertex_normals_vao = None
        self.vertex_normals_vbo = None
        self.vertex_normals_count = 0

        # Point Cloud Buffer (points)
        self.point_cloud_vao = None
        self.point_cloud_vbo = None
        self.point_cloud_count = 0

        # Point Cloud Normals Buffer (lines)
        self.point_cloud_normals_vao = None
        self.point_cloud_normals_vbo = None
        self.point_cloud_normals_count = 0

        # Non-manifold Edges Buffer (lines)
        self.nonmanifold_edges_vao = None
        self.nonmanifold_edges_vbo = None
        self.nonmanifold_edges_count = 0

        # Internal Edges Buffer (lines)
        self.internal_edges_vao = None
        self.internal_edges_vbo = None
        self.internal_edges_count = 0

        # Boundary Edges Buffer (lines)
        self.boundary_edges_vao = None
        self.boundary_edges_vbo = None
        self.boundary_edges_count = 0

        # Non-manifold Vertices Buffer (points)
        self.nonmanifold_vertices_vao = None
        self.nonmanifold_vertices_vbo = None
        self.nonmanifold_vertices_count = 0

    def update_from_mesh(
        self, 
        mesh: trimesh.Trimesh, 
        mesh_info: MeshInfo, 
        normal_length: float, 
        points: np.ndarray, 
        point_normals: np.ndarray,
        program: moderngl.Program
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
        self.update_gpu_buffers(program)

    def release(self):
        """Release all ModernGL objects."""
        for attr in [
            'main_vao', 'main_vbo', 'main_ebo',
            'intersected_vao', 'intersected_vbo', 'intersected_ebo',
            'face_normals_vao', 'face_normals_vbo',
            'vertex_normals_vao', 'vertex_normals_vbo',
            'point_cloud_vao', 'point_cloud_vbo',
            'point_cloud_normals_vao', 'point_cloud_normals_vbo',
            'nonmanifold_edges_vao', 'nonmanifold_edges_vbo',
            'internal_edges_vao', 'internal_edges_vbo',
            'boundary_edges_vao', 'boundary_edges_vbo',
            'nonmanifold_vertices_vao', 'nonmanifold_vertices_vbo'
        ]:
            obj = getattr(self, attr)
            if obj:
                obj.release()
                setattr(self, attr, None)

    def update_gpu_buffers(self, program: moderngl.Program):
        # Release old buffers first
        self.release()

        # Split faces into two groups
        all_indices = np.arange(len(self.mesh.faces))
        intersected_mask = np.array([i in self.intersected_face_ids for i in all_indices])

        # 1. Prepare Main Mesh (Not highlighted)
        main_faces = self.mesh.faces
        self.main_vao, self.main_vbo, self.main_ebo, self.main_index_count = self.setup_buffer(
            program, 
            main_faces
        )

        # 2. Prepare Intersected Faces (Selected)
        intersected_faces = self.mesh.faces[intersected_mask]
        self.intersected_vao, self.intersected_vbo, self.intersected_ebo, self.intersected_index_count = self.setup_buffer(
            program,
            intersected_faces,
        )

        # 3. Prepare Normals
        self.face_normals_vao, self.face_normals_vbo, self.face_normals_count = self.setup_face_normals_buffer(program)
        self.vertex_normals_vao, self.vertex_normals_vbo, self.vertex_normals_count = self.setup_vertex_normals_buffer(program)

        # 4. Prepare Point Cloud
        (self.point_cloud_vao, self.point_cloud_vbo, self.point_cloud_count, 
         self.point_cloud_normals_vao, self.point_cloud_normals_vbo, self.point_cloud_normals_count) = self.setup_point_cloud_buffer(program)

        # 5. Prepare Non-manifold Edges
        self.nonmanifold_edges_vao, self.nonmanifold_edges_vbo, self.nonmanifold_edges_count = self.setup_nonmanifold_edges_buffer(program)

        # 6. Prepare Non-manifold Vertices
        self.nonmanifold_vertices_vao, self.nonmanifold_vertices_vbo, self.nonmanifold_vertices_count = self.setup_nonmanifold_vertices_buffer(program)

        # 7. Prepare Internal and Boundary Edges
        self.internal_edges_vao, self.internal_edges_vbo, self.internal_edges_count = self.setup_internal_edges_buffer(program)
        self.boundary_edges_vao, self.boundary_edges_vbo, self.boundary_edges_count = self.setup_boundary_edges_buffer(program)

    def setup_buffer(self, program, faces):
        if len(faces) == 0:
            return None, None, None, 0

        # Un-index vertices so each triangle has unique data.
        vertices = self.mesh.vertices[faces].reshape(-1, 3)
        data = vertices.astype(np.float32).tobytes()
        indices = np.arange(len(vertices)).astype(np.uint32).tobytes()

        vbo = self.ctx.buffer(data)
        ebo = self.ctx.buffer(indices)

        # Layout: Pos(3)
        vao = self.ctx.vertex_array(program, [(vbo, '3f', 'aPos')], index_buffer=ebo)

        return vao, vbo, ebo, len(vertices)

    def setup_point_cloud_buffer(self, program):
        points = self.points
        data = points.astype(np.float32).tobytes()

        vbo = self.ctx.buffer(data)
        vao = self.ctx.vertex_array(program, [(vbo, '3f', 'aPos')])

        # Prepare Point Cloud Normals
        normals = self.point_normals
        line_verts = np.empty((points.shape[0] * 2, 3), dtype=np.float32)
        line_verts[0::2] = points
        line_verts[1::2] = points + normals * self.normal_length

        data_normals = line_verts.astype(np.float32).tobytes()
        vbo_normals = self.ctx.buffer(data_normals)
        vao_normals = self.ctx.vertex_array(program, [(vbo_normals, '3f', 'aPos')])

        return vao, vbo, points.shape[0], vao_normals, vbo_normals, line_verts.shape[0]

    def setup_face_normals_buffer(self, program):
        # Face centers and normals
        centers = self.mesh.triangles_center
        normals = self.mesh.face_normals

        line_verts = np.empty((centers.shape[0] * 2, 3), dtype=np.float32)
        line_verts[0::2] = centers
        line_verts[1::2] = centers + normals * self.normal_length

        data = line_verts.astype(np.float32).tobytes()
        vbo = self.ctx.buffer(data)
        vao = self.ctx.vertex_array(program, [(vbo, '3f', 'aPos')])

        return vao, vbo, line_verts.shape[0]

    def setup_vertex_normals_buffer(self, program):
        verts = self.mesh.vertices
        normals = self.mesh.vertex_normals

        line_verts = np.empty((verts.shape[0] * 2, 3), dtype=np.float32)
        line_verts[0::2] = verts
        line_verts[1::2] = verts + normals * self.normal_length

        data = line_verts.astype(np.float32).tobytes()
        vbo = self.ctx.buffer(data)
        vao = self.ctx.vertex_array(program, [(vbo, '3f', 'aPos')])

        return vao, vbo, line_verts.shape[0]

    def setup_nonmanifold_edges_buffer(self, program):
        if len(self.mesh_info.nonmanifold_edges) == 0: return None, None, 0

        nonmanifold_edges = self.mesh_info.nonmanifold_edges
        verts = self.mesh.vertices

        line_verts = np.empty((nonmanifold_edges.shape[0] * 2, 3), dtype=np.float32)
        line_verts[0::2] = verts[nonmanifold_edges[:, 0]]
        line_verts[1::2] = verts[nonmanifold_edges[:, 1]]

        data = line_verts.astype(np.float32).tobytes()
        vbo = self.ctx.buffer(data)
        vao = self.ctx.vertex_array(program, [(vbo, '3f', 'aPos')])

        return vao, vbo, line_verts.shape[0]

    def setup_nonmanifold_vertices_buffer(self, program):
        if len(self.mesh_info.nonmanifold_vertices) == 0: return None, None, 0

        nonmanifold_vertices = self.mesh_info.nonmanifold_vertices
        verts = self.mesh.vertices
        vertex_positions = verts[nonmanifold_vertices]

        data = vertex_positions.astype(np.float32).tobytes()
        vbo = self.ctx.buffer(data)
        vao = self.ctx.vertex_array(program, [(vbo, '3f', 'aPos')])

        return vao, vbo, vertex_positions.shape[0]

    def setup_internal_edges_buffer(self, program):
        if len(self.mesh_info.internal_edges) == 0: return None, None, 0
        edges = self.mesh_info.internal_edges
        verts = self.mesh.vertices
        line_verts = np.empty((edges.shape[0] * 2, 3), dtype=np.float32)
        line_verts[0::2] = verts[edges[:, 0]]
        line_verts[1::2] = verts[edges[:, 1]]
        data = line_verts.astype(np.float32).tobytes()
        vbo = self.ctx.buffer(data)
        vao = self.ctx.vertex_array(program, [(vbo, '3f', 'aPos')])
        return vao, vbo, line_verts.shape[0]

    def setup_boundary_edges_buffer(self, program):
        if len(self.mesh_info.boundary_edges) == 0: return None, None, 0
        edges = self.mesh_info.boundary_edges
        verts = self.mesh.vertices
        line_verts = np.empty((edges.shape[0] * 2, 3), dtype=np.float32)
        line_verts[0::2] = verts[edges[:, 0]]
        line_verts[1::2] = verts[edges[:, 1]]
        data = line_verts.astype(np.float32).tobytes()
        vbo = self.ctx.buffer(data)
        vao = self.ctx.vertex_array(program, [(vbo, '3f', 'aPos')])
        return vao, vbo, line_verts.shape[0]
