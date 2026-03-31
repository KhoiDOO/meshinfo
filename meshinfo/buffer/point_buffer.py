import ctypes

import numpy as np
from OpenGL.GL import *

from ..analysis.pc import PointInfo
from ..constants import VERTEX_STRIDE


class PointBuffer:
    def __init__(self):
        self.points: np.ndarray = None
        self.normals: np.ndarray = None
        self.normal_length = 0.0
        self.bounds = None
        self.bounds_center = None
        self.bounds_size = None
        self.original_bounds = None
        self.original_bounds_center = None
        self.original_bounds_size = None
        self.position = np.zeros(3, dtype=np.float32)
        
        # Main Buffer
        self.point_vao = glGenVertexArrays(1)
        self.point_vbo = glGenBuffers(1)
        self.point_cloud_count = 0

        # Point Cloud Normals
        self.normal_vao = glGenVertexArrays(1)
        self.normal_vbo = glGenBuffers(1)
        self.normal_count = 0
    
    def update_point_cloud(self, points: np.ndarray, point_info: PointInfo, normal_length, color_scheme, normals: np.ndarray = None):

        self.points = points
        self.normals = normals
        self.normal_length = normal_length

        self.point_cloud_count = len(points)
        self.normal_count = len(normals) if normals is not None else 0

        self.bounds = point_info.analysis["bounds"]
        self.bounds_center = (self.bounds[0] + self.bounds[1]) * 0.5
        self.bounds_size = self.bounds[1] - self.bounds[0]

        self.original_bounds = np.copy(self.bounds)
        self.original_bounds_center = np.copy(self.bounds_center)
        self.original_bounds_size = np.copy(self.bounds_size)

        self.setup_point_cloud_buffer(color_scheme)
    
    def refresh_colors(self, color_scheme):
        # Colors are provided by uniforms at draw time.
        return

    def setup_point_cloud_buffer(self, color_scheme):
        points = self.points
        data = points.astype(np.float32)

        glBindVertexArray(self.point_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.point_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

        # Prepare Point Cloud Normals
        if self.normals is None:
            return points.shape[0], 0
        
        normals = self.normals
        line_verts = np.empty((points.shape[0] * 2, 3), dtype=np.float32)
        line_verts[0::2] = points
        line_verts[1::2] = points + normals * self.normal_length

        data = line_verts.astype(np.float32)

        glBindVertexArray(self.normal_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.normal_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

        return points.shape[0], line_verts.shape[0]