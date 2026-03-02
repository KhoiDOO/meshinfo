import os

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

import numpy as np
from pyrr import Matrix44
import math

from PIL import Image

from utils.io import load_mesh
from utils.fdialog import open_file_dialog as show_open_file_dialog, save_file_dialog as show_save_file_dialog
from analysis.mesh import MeshInfo
from buffer.mesh_buffer import MeshBuffer

from constants import *

class MeshViewer:
    def __init__(self):
        self.mode = DEFAULT_MODE
        self.mesh_buffers: list[MeshBuffer] = []
        
        self.show_intersected = DEFAULT_SHOW_INTERSECTED
        self.show_face_normals = DEFAULT_SHOW_FACE_NORMALS
        self.show_vertex_normals = DEFAULT_SHOW_VERTEX_NORMALS
        self.show_point_cloud = DEFAULT_SHOW_POINT_CLOUD
        self.show_point_cloud_normals = DEFAULT_SHOW_POINT_CLOUD_NORMALS
        self.show_nonmanifold_edges = DEFAULT_SHOW_NONMANIFOLD_EDGES
        self.show_nonmanifold_vertices = DEFAULT_SHOW_NONMANIFOLD_VERTICES

        self.color_theme = DEFAULT_COLOR_THEME

        # Camera control
        self.camera_rotating = DEFAULT_CAMERA_ROTATING
        self.camera_angle = DEFAULT_CAMERA_ANGLE
        self.camera_vertical_angle = DEFAULT_CAMERA_VERTICAL_ANGLE
        self.camera_distance = DEFAULT_CAMERA_DISTANCE
        self.camera_height = DEFAULT_CAMERA_HEIGHT
        self.camera_rotation_speed = DEFAULT_CAMERA_ROTATION_SPEED
        self.camera_manual_speed = DEFAULT_CAMERA_MANUAL_SPEED
        self.camera_height_speed = DEFAULT_CAMERA_HEIGHT_SPEED

        # Object control
        self.object_rotation_x = DEFAULT_OBJECT_ROTATION_X
        self.object_rotation_y = DEFAULT_OBJECT_ROTATION_Y
        self.object_rotation_z = DEFAULT_OBJECT_ROTATION_Z
        self.object_rotation_speed = DEFAULT_OBJECT_ROTATION_SPEED
        self.object_scale = DEFAULT_OBJECT_SCALE
        self.object_scale_speed = DEFAULT_OBJECT_SCALE_SPEED
        self.mesh_layout_padding = MESH_LAYOUT_PADDING

        self.last_o_state = glfw.RELEASE
        self.last_tab_state = glfw.RELEASE
        self.last_i_state = glfw.RELEASE
        self.last_j_state = glfw.RELEASE
        self.last_k_state = glfw.RELEASE
        self.last_l_state = glfw.RELEASE
        self.last_n_state = glfw.RELEASE
        self.last_m_state = glfw.RELEASE
        self.last_p_state = glfw.RELEASE
        self.last_y_state = glfw.RELEASE
        self.last_c_state = glfw.RELEASE
        self.last_u_state = glfw.RELEASE
        self.last_h_state = glfw.RELEASE
        self.last_v_state = glfw.RELEASE
        self.last_space_state = glfw.RELEASE
        self.last_r_state = glfw.RELEASE
        self.last_left_bracket_state = glfw.RELEASE
        self.last_right_bracket_state = glfw.RELEASE

        if not glfw.init():
            raise Exception("GLFW could not be initialized!")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        self.window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window could not be created!")

        glfw.make_context_current(self.window)
        glEnable(GL_DEPTH_TEST)
        
        # Create and bind a VAO before shader compilation (required for macOS Core Profile)
        dummy_vao = glGenVertexArrays(1)
        glBindVertexArray(dummy_vao)
        
        self.shader = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        
        # Enable programmatically set point sizes
        glEnable(GL_PROGRAM_POINT_SIZE)

        self.mvp_loc = glGetUniformLocation(self.shader, "mvp")
        self.override_loc = glGetUniformLocation(self.shader, "overrideColor")
        self.use_override_loc = glGetUniformLocation(self.shader, "useOverride")
        self.point_size_loc = glGetUniformLocation(self.shader, "pointSize")

    def open_file_dialog(self, renew_buffers=True):
        file_paths = show_open_file_dialog(
            DIALOG_TITLE_SELECT_MESH,
            MESH_FILE_TYPES,
            allow_multiple=True,
        )

        if renew_buffers:
            self.mesh_buffers = []

        if file_paths:
            self.load_mesh(file_paths)
    
    def capture_screenshot(self):
        """Capture the mesh area and save it."""
        width, height = glfw.get_window_size(self.window)
        
        # Read full framebuffer
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # Convert to PIL Image and flip vertically
        image_data = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        image_data = np.flipud(image_data)
        img = Image.fromarray(image_data, 'RGB')
        
        # Autocrop to remove empty space (black background)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        
        file_path = show_save_file_dialog(
            DIALOG_TITLE_SAVE_SCREENSHOT,
            SCREENSHOT_DEFAULT_EXTENSION,
            SCREENSHOT_FILE_TYPES,
            default_name="screenshot",
        )
        
        if file_path:
            try:
                if file_path.lower().endswith('.pdf'):
                    img.save(file_path, 'PDF')
                    print(f"PDF saved: {file_path}")
                else:
                    img.save(file_path)
                    print(f"Screenshot saved: {file_path}")
            except Exception as e:
                print(f"Failed to save screenshot: {e}")

    def get_color_scheme(self):
        """Return color scheme based on current theme."""
        if self.color_theme == THEME_DARK:
            return COLOR_SCHEME_DARK
        else:
            return COLOR_SCHEME_LIGHT

    def load_mesh(self, path):
        if isinstance(path, (list, tuple)):
            for item in path:
                self.load_single_mesh(item)
            self.layout_meshes()
            return
        try:
            self.load_single_mesh(path)
            self.layout_meshes()
            
        except Exception as e:
            print(f"Failed to load mesh: {e}")

    def load_single_mesh(self, path:str):
        mesh = load_mesh(path)
        if mesh is None:
            print("Mesh is None after loading. Check if the file is valid and supported.")
            return
        if len(mesh.faces) == 0:
            print("Loaded mesh has no faces. Please select a valid mesh file.")
            return
        if len(mesh.vertices) == 0:
            print("Loaded mesh has no vertices. Please select a valid mesh file.")
            return
        # mesh analysis and info extraction
        name = os.path.basename(path).split('.')[0]
        mesh_info = MeshInfo(mesh, name=name)

        # Calculate diagonal and normal length for visualization
        bounds = mesh_info.analysis["bounds"]
        diag = np.linalg.norm(bounds[1] - bounds[0])
        normal_length = max(diag * NORMAL_LENGTH_FACTOR, NORMAL_LENGTH_MIN)

        # Sample points for point cloud visualization (if needed)
        points, face_idx = mesh.sample(POINT_CLOUD_SAMPLE_COUNT, return_index=True)
        point_normals = mesh.face_normals[face_idx]

        mesh_buffer = MeshBuffer()
        mesh_buffer.update_from_mesh(
            mesh,
            mesh_info,
            normal_length,
            points,
            point_normals,
            self.get_color_scheme(),
        )
        self.mesh_buffers.append(mesh_buffer)

        print(mesh_info)

    def layout_meshes(self):
        if not self.mesh_buffers:
            return

        count = len(self.mesh_buffers)
        grid_cols = int(math.ceil(math.sqrt(count)))
        grid_rows = int(math.ceil(count / grid_cols))

        # Use original bounds for layout (unaffected by object scale)
        max_extent = 0.0
        for buffer in self.mesh_buffers:
            if buffer.original_bounds_size is not None:
                max_extent = max(max_extent, float(np.max(buffer.original_bounds_size)))

        if max_extent <= 0.0:
            max_extent = 1.0

        # Scale spacing by object_scale so spacing adjusts proportionally with mesh size
        spacing = max_extent * (1.0 + self.mesh_layout_padding) * self.object_scale

        for index, buffer in enumerate(self.mesh_buffers):
            row = index // grid_cols
            col = index % grid_cols

            grid_x = (col - (grid_cols - 1) * 0.5) * spacing
            grid_z = (row - (grid_rows - 1) * 0.5) * spacing
            center = buffer.original_bounds_center if buffer.original_bounds_center is not None else np.zeros(3, dtype=np.float32)
            buffer.position = np.array([grid_x - center[0], -center[1], grid_z - center[2]], dtype=np.float32)
    
    def handle_input(self):

        # Handle 'I' for toggling intersected faces
        i_state = glfw.get_key(self.window, glfw.KEY_I)
        if i_state == glfw.PRESS and self.last_i_state == glfw.RELEASE:
            self.show_intersected = not self.show_intersected
        self.last_i_state = i_state

        # Handle 'J' for Mode 0 (SOLID)
        j_state = glfw.get_key(self.window, glfw.KEY_J)
        if j_state == glfw.PRESS and self.last_j_state == glfw.RELEASE:
            self.mode = MODE_SOLID
        self.last_j_state = j_state

        # Handle 'K' for Mode 1 (WIREFRAME)
        k_state = glfw.get_key(self.window, glfw.KEY_K)
        if k_state == glfw.PRESS and self.last_k_state == glfw.RELEASE:
            self.mode = MODE_WIREFRAME
        self.last_k_state = k_state

        # Handle 'L' for Mode 2 (BOTH)
        l_state = glfw.get_key(self.window, glfw.KEY_L)
        if l_state == glfw.PRESS and self.last_l_state == glfw.RELEASE:
            self.mode = MODE_BOTH
        self.last_l_state = l_state

        # Handle 'N' for per-face normals
        n_state = glfw.get_key(self.window, glfw.KEY_N)
        if n_state == glfw.PRESS and self.last_n_state == glfw.RELEASE:
            self.show_face_normals = not self.show_face_normals
        self.last_n_state = n_state

        # Handle 'M' for per-vertex normals
        m_state = glfw.get_key(self.window, glfw.KEY_M)
        if m_state == glfw.PRESS and self.last_m_state == glfw.RELEASE:
            self.show_vertex_normals = not self.show_vertex_normals
        self.last_m_state = m_state

        # Handle 'P' for point cloud
        p_state = glfw.get_key(self.window, glfw.KEY_P)
        if p_state == glfw.PRESS and self.last_p_state == glfw.RELEASE:
            self.show_point_cloud = not self.show_point_cloud
        self.last_p_state = p_state

        # Handle 'Y' for point cloud normals
        y_state = glfw.get_key(self.window, glfw.KEY_Y)
        if y_state == glfw.PRESS and self.last_y_state == glfw.RELEASE:
            self.show_point_cloud_normals = not self.show_point_cloud_normals
        self.last_y_state = y_state

        # Handle 'H' for non-manifold edges
        h_state = glfw.get_key(self.window, glfw.KEY_H)
        if h_state == glfw.PRESS and self.last_h_state == glfw.RELEASE:
            self.show_nonmanifold_edges = not self.show_nonmanifold_edges
        self.last_h_state = h_state

        # Handle 'V' for non-manifold vertices
        v_state = glfw.get_key(self.window, glfw.KEY_V)
        if v_state == glfw.PRESS and self.last_v_state == glfw.RELEASE:
            self.show_nonmanifold_vertices = not self.show_nonmanifold_vertices
        self.last_v_state = v_state

        # Handle 'O' for Open
        o_state = glfw.get_key(self.window, glfw.KEY_O)
        if o_state == glfw.PRESS and self.last_o_state == glfw.RELEASE:
            self.open_file_dialog()
        self.last_o_state = o_state

        # Handle "Tab" for Open without renewing buffers
        tab_state = glfw.get_key(self.window, glfw.KEY_TAB)
        if tab_state == glfw.PRESS and self.last_tab_state == glfw.RELEASE:
            self.open_file_dialog(renew_buffers=False)
        self.last_tab_state = tab_state

        # Handle 'C' for Capture Screenshot
        c_state = glfw.get_key(self.window, glfw.KEY_C)
        if c_state == glfw.PRESS and self.last_c_state == glfw.RELEASE:
            self.capture_screenshot()
        self.last_c_state = c_state

        # Handle 'U' for toggle color theme
        u_state = glfw.get_key(self.window, glfw.KEY_U)
        if u_state == glfw.PRESS and self.last_u_state == glfw.RELEASE:
            self.color_theme = THEME_LIGHT if self.color_theme == THEME_DARK else THEME_DARK
            if self.mesh_buffers:
                colors_scheme = self.get_color_scheme()
                for buffer in self.mesh_buffers:
                    buffer.refresh_colors(colors_scheme)
        self.last_u_state = u_state

        # Handle [ and ] for layout spacing
        left_bracket_state = glfw.get_key(self.window, glfw.KEY_LEFT_BRACKET)
        if left_bracket_state == glfw.PRESS and self.last_left_bracket_state == glfw.RELEASE:
            self.mesh_layout_padding = max(
                MESH_LAYOUT_PADDING_MIN,
                self.mesh_layout_padding - MESH_LAYOUT_PADDING_STEP,
            )
            self.layout_meshes()
        self.last_left_bracket_state = left_bracket_state

        right_bracket_state = glfw.get_key(self.window, glfw.KEY_RIGHT_BRACKET)
        if right_bracket_state == glfw.PRESS and self.last_right_bracket_state == glfw.RELEASE:
            self.mesh_layout_padding = min(
                MESH_LAYOUT_PADDING_MAX,
                self.mesh_layout_padding + MESH_LAYOUT_PADDING_STEP,
            )
            self.layout_meshes()
        self.last_right_bracket_state = right_bracket_state

        # Handle SPACE for toggling camera rotation
        space_state = glfw.get_key(self.window, glfw.KEY_SPACE)
        if space_state == glfw.PRESS and self.last_space_state == glfw.RELEASE:
            self.camera_rotating = not self.camera_rotating
        self.last_space_state = space_state

        # Handle R for reset camera
        r_state = glfw.get_key(self.window, glfw.KEY_R)
        if r_state == glfw.PRESS and self.last_r_state == glfw.RELEASE:
            self.camera_rotating = DEFAULT_CAMERA_ROTATING
            self.camera_angle = DEFAULT_CAMERA_ANGLE
            self.camera_vertical_angle = DEFAULT_CAMERA_VERTICAL_ANGLE
            self.camera_distance = DEFAULT_CAMERA_DISTANCE
            self.camera_height = DEFAULT_CAMERA_HEIGHT
        self.last_r_state = r_state

        # Handle object rotation and zoom (always available)
        delta_time = DELTA_TIME
        rotation_step = self.object_rotation_speed * delta_time
        scale_step = self.object_scale_speed * delta_time
        height_step = self.camera_height_speed * delta_time

        # A/D: Rotate object left/right (Y axis)
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.object_rotation_y += rotation_step
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.object_rotation_y -= rotation_step

        # W/S: Rotate object up/down (X axis)
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.object_rotation_x += rotation_step
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.object_rotation_x -= rotation_step

        # Q/E: Roll object (Z axis)
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            self.object_rotation_z += rotation_step
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            self.object_rotation_z -= rotation_step

        # Z/X: Scale object down/up
        if glfw.get_key(self.window, glfw.KEY_Z) == glfw.PRESS:
            self.object_scale = max(OBJECT_SCALE_MIN, self.object_scale - scale_step)
        if glfw.get_key(self.window, glfw.KEY_X) == glfw.PRESS:
            self.object_scale = min(OBJECT_SCALE_MAX, self.object_scale + scale_step)

        # Up/Down: Move camera vertically
        if glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS:
            self.camera_height = min(CAMERA_HEIGHT_MAX, self.camera_height + height_step)
        if glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS:
            self.camera_height = max(CAMERA_HEIGHT_MIN, self.camera_height - height_step)
    
    def render_mesh(self):
        glUseProgram(self.shader)
        colors_scheme = self.get_color_scheme()

        proj = Matrix44.perspective_projection(CAMERA_FOV, WINDOW_WIDTH/WINDOW_HEIGHT, CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE)
        
        # Calculate camera position (horizontal orbit only)
        if self.camera_rotating:
            angle = glfw.get_time() * self.camera_rotation_speed
        else:
            angle = self.camera_angle
        vertical_angle = self.camera_vertical_angle
        
        # Calculate camera position with both horizontal and vertical rotation
        horizontal_distance = math.cos(vertical_angle) * self.camera_distance
        cam_x = math.sin(angle) * horizontal_distance
        cam_z = math.cos(angle) * horizontal_distance
        cam_y = self.camera_height
        view = Matrix44.look_at([cam_x, cam_y, cam_z], [0, 0, 0], [0, 1, 0])
        
        for buffer in self.mesh_buffers:
            model = (
                Matrix44.from_translation(buffer.position)
                * Matrix44.from_x_rotation(self.object_rotation_x)
                * Matrix44.from_y_rotation(self.object_rotation_y)
                * Matrix44.from_z_rotation(self.object_rotation_z)
                * Matrix44.from_scale([self.object_scale, self.object_scale, self.object_scale])
            )
            mvp = proj * view * model
            glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, mvp)

            # --- DRAW PASS 1: Main Mesh ---
            if buffer.main_index_count > 0:
                glBindVertexArray(buffer.main_vao)
                glUniform1i(self.use_override_loc, False)

                if self.mode == MODE_SOLID or self.mode == MODE_BOTH:
                    # Add offset to push solid faces away from lines/highlighter
                    glEnable(GL_POLYGON_OFFSET_FILL)
                    glPolygonOffset(POLYGON_OFFSET_FACTOR, POLYGON_OFFSET_UNITS)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                    glDrawElements(GL_TRIANGLES, buffer.main_index_count, GL_UNSIGNED_INT, None)
                    glDisable(GL_POLYGON_OFFSET_FILL)

                if self.mode == MODE_WIREFRAME or self.mode == MODE_BOTH:
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    glUniform1i(self.use_override_loc, True)
                    glUniform3f(self.override_loc, *colors_scheme['wireframe'])
                    glDrawElements(GL_TRIANGLES, buffer.main_index_count, GL_UNSIGNED_INT, None)

            # --- DRAW PASS 2: Highlighted Part ---
            if self.show_intersected and buffer.intersected_index_count > 0:
                glBindVertexArray(buffer.intersected_vao)

                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glUniform1i(self.use_override_loc, True)
                glUniform3f(self.override_loc, *colors_scheme['intersected'])

                glEnable(GL_POLYGON_OFFSET_FILL)
                glPolygonOffset(POLYGON_OFFSET_FACTOR, POLYGON_OFFSET_UNITS)
                glDrawElements(GL_TRIANGLES, buffer.intersected_index_count, GL_UNSIGNED_INT, None)
                glDisable(GL_POLYGON_OFFSET_FILL)

                # 2. Wireframe Outline for Highlight
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glUniform1i(self.use_override_loc, True)  # Keep override on for outline color
                glUniform3f(self.override_loc, *colors_scheme['wireframe_highlight'])
                glDrawElements(GL_TRIANGLES, buffer.intersected_index_count, GL_UNSIGNED_INT, None)

            # --- DRAW PASS 3: Normals ---
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glUniform1i(self.use_override_loc, True)

            if self.show_face_normals and buffer.face_normals_count > 0:
                glBindVertexArray(buffer.face_normals_vao)
                glUniform3f(self.override_loc, *colors_scheme['face_normals'])
                glDrawArrays(GL_LINES, 0, buffer.face_normals_count)

            if self.show_vertex_normals and buffer.vertex_normals_count > 0:
                glBindVertexArray(buffer.vertex_normals_vao)
                glUniform3f(self.override_loc, *colors_scheme['vertex_normals'])
                glDrawArrays(GL_LINES, 0, buffer.vertex_normals_count)

            if self.show_point_cloud and buffer.point_cloud_count > 0:
                glBindVertexArray(buffer.point_cloud_vao)
                glUniform3f(self.override_loc, *colors_scheme['point_cloud'])
                glUniform1f(self.point_size_loc, POINT_CLOUD_POINT_SIZE)
                glDrawArrays(GL_POINTS, 0, buffer.point_cloud_count)

            if self.show_point_cloud_normals and buffer.point_cloud_normals_count > 0:
                glBindVertexArray(buffer.point_cloud_normals_vao)
                glUniform3f(self.override_loc, *colors_scheme['point_cloud_normals'])
                glDrawArrays(GL_LINES, 0, buffer.point_cloud_normals_count)

            if self.show_nonmanifold_edges and buffer.nonmanifold_edges_count > 0:
                # Render non-manifold edges on top without depth testing
                glDisable(GL_DEPTH_TEST)
                glBindVertexArray(buffer.nonmanifold_edges_vao)
                glUniform3f(self.override_loc, *colors_scheme['nonmanifold_edges'])
                glDrawArrays(GL_LINES, 0, buffer.nonmanifold_edges_count)
                glEnable(GL_DEPTH_TEST)

            if self.show_nonmanifold_vertices and buffer.nonmanifold_vertices_count > 0:
                # Render non-manifold vertices as points on top
                glDisable(GL_DEPTH_TEST)
                glBindVertexArray(buffer.nonmanifold_vertices_vao)
                glUniform3f(self.override_loc, *colors_scheme['nonmanifold_vertices'])
                glUniform1f(self.point_size_loc, NONMANIFOLD_VERTEX_POINT_SIZE)
                glDrawArrays(GL_POINTS, 0, buffer.nonmanifold_vertices_count)
                glEnable(GL_DEPTH_TEST)

    def run(self):
        while not glfw.window_should_close(self.window):
            self.handle_input()
            
            colors_scheme = self.get_color_scheme()
            glClearColor(*colors_scheme['background'])
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            if self.mesh_buffers:
                self.render_mesh()

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

if __name__ == "__main__":
    app = MeshViewer()
    app.run()