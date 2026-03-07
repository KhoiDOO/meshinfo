import os

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

import numpy as np
from pyrr import Matrix44
import math

from meshinfo.utils.io import load_pc
from meshinfo.utils.fdialog import open_file_dialog as show_open_file_dialog
from meshinfo.buffer.point_buffer import PointBuffer
from meshinfo.analysis.pc import PointInfo

from meshinfo.constants import *

class PointViewer:
    def __init__(self):
        self.point_buffers: list[PointBuffer] = []

        self.show_point_normals = DEFAULT_SHOW_POINT_CLOUD_NORMALS
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
        self.pc_layout_padding = MESH_LAYOUT_PADDING

        self.last_o_state = glfw.RELEASE
        self.last_tab_state = glfw.RELEASE
        self.last_p_state = glfw.RELEASE
        self.last_u_state = glfw.RELEASE
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
            DIALOG_TITLE_SELECT_POINT_CLOUD,
            POINT_CLOUD_FILE_TYPES,
            allow_multiple=True,
        )

        if renew_buffers:
            self.point_buffers = []

        if file_paths:
            self.load_pc(file_paths)
    
    def get_color_scheme(self):
        """Return color scheme based on current theme."""
        if self.color_theme == THEME_DARK:
            return COLOR_SCHEME_DARK
        else:
            return COLOR_SCHEME_LIGHT
    
    def load_pc(self, path):
        if isinstance(path, (list, tuple)):
            for item in path:
                self.load_single_pc(item)
            self.layout_pcs()
            return
        try:
            self.load_single_pc(path)
            self.layout_pcs()
            
        except Exception as e:
            print(f"Failed to load point cloud: {e}")
    
    def load_single_pc(self, path):
        points, normals = load_pc(path)

        if len(points) == 0:
            print(f"Warning: No points found in {path}")
            return
    
        name = os.path.basename(path).split('.')[0]
        point_info = PointInfo(points, normals, name)

        # Calculate diagonal and normal length for visualization
        bounds = point_info.analysis["bounds"]
        diag = np.linalg.norm(bounds[1] - bounds[0])
        normal_length = max(diag * NORMAL_LENGTH_FACTOR, NORMAL_LENGTH_MIN)

        point_buffer = PointBuffer()
        point_buffer.update_point_cloud(
            points, 
            point_info, 
            normal_length, 
            self.get_color_scheme(), 
            normals,
        )
        self.point_buffers.append(point_buffer)

        print(point_info)
    
    def layout_pcs(self):

        if not self.point_buffers:
            return

        count = len(self.point_buffers)
        grid_cols = int(math.ceil(math.sqrt(count)))
        grid_rows = int(math.ceil(count / grid_cols))

        # Use original bounds for layout (unaffected by object scale)
        max_extent = 0.0
        for buffer in self.point_buffers:
            if buffer.original_bounds_size is not None:
                max_extent = max(max_extent, float(np.max(buffer.original_bounds_size)))
        
        if max_extent <= 0.0:
            max_extent = 1.0
        
        # Scale spacing by object_scale so spacing adjusts proportionally with mesh size
        spacing = max_extent * (1.0 + self.pc_layout_padding) * self.object_scale

        for index, buffer in enumerate(self.point_buffers):
            row = index // grid_cols
            col = index % grid_cols

            grid_x = (col - (grid_cols - 1) * 0.5) * spacing
            grid_z = (row - (grid_rows - 1) * 0.5) * spacing
            center = buffer.original_bounds_center if buffer.original_bounds_center is not None else np.zeros(3, dtype=np.float32)
            buffer.position = np.array([grid_x - center[0], -center[1], grid_z - center[2]], dtype=np.float32)
    
    def handle_input(self):
        
        # Handle 'Y' for point cloud normals
        y_state = glfw.get_key(self.window, glfw.KEY_Y)
        if y_state == glfw.PRESS and self.last_y_state == glfw.RELEASE:
            self.show_point_normals = not self.show_point_normals
        self.last_y_state = y_state

        # Handle 'O' for Open
        o_state = glfw.get_key(self.window, glfw.KEY_O)
        if o_state == glfw.PRESS and self.last_o_state == glfw.RELEASE:
            self.open_file_dialog(renew_buffers=True)
        self.last_o_state = o_state

        # Handle "Tab" for Open without renewing buffers
        tab_state = glfw.get_key(self.window, glfw.KEY_TAB)
        if tab_state == glfw.PRESS and self.last_tab_state == glfw.RELEASE:
            self.open_file_dialog(renew_buffers=False)
        self.last_tab_state = tab_state

        # Handle 'U' for toggle color theme
        u_state = glfw.get_key(self.window, glfw.KEY_U)
        if u_state == glfw.PRESS and self.last_u_state == glfw.RELEASE:
            self.color_theme = THEME_LIGHT if self.color_theme == THEME_DARK else THEME_DARK
            if self.point_buffers:
                colors_scheme = self.get_color_scheme()
                for buffer in self.point_buffers:
                    buffer.refresh_colors(colors_scheme)
        self.last_u_state = u_state

        # Handle [ and ] for layout spacing
        left_bracket_state = glfw.get_key(self.window, glfw.KEY_LEFT_BRACKET)
        if left_bracket_state == glfw.PRESS and self.last_left_bracket_state == glfw.RELEASE:
            self.pc_layout_padding = max(
                PC_LAYOUT_PADDING_MIN,
                self.pc_layout_padding - PC_LAYOUT_PADDING_STEP,
            )
            self.layout_pcs()
        self.last_left_bracket_state = left_bracket_state

        right_bracket_state = glfw.get_key(self.window, glfw.KEY_RIGHT_BRACKET)
        if right_bracket_state == glfw.PRESS and self.last_right_bracket_state == glfw.RELEASE:
            self.pc_layout_padding = min(
                PC_LAYOUT_PADDING_MAX,
                self.pc_layout_padding + PC_LAYOUT_PADDING_STEP,
            )
            self.layout_pcs()
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
    
    def render_pc(self):
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
        
        for buffer in self.point_buffers:
            model = (
                Matrix44.from_translation(buffer.position)
                * Matrix44.from_x_rotation(self.object_rotation_x)
                * Matrix44.from_y_rotation(self.object_rotation_y)
                * Matrix44.from_z_rotation(self.object_rotation_z)
                * Matrix44.from_scale([self.object_scale, self.object_scale, self.object_scale])
            )
            mvp = proj * view * model
            glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, mvp)

            glBindVertexArray(buffer.point_vao)
            glUniform3f(self.override_loc, *colors_scheme['point_cloud'])
            glUniform1f(self.point_size_loc, POINT_CLOUD_POINT_SIZE)
            glDrawArrays(GL_POINTS, 0, buffer.point_cloud_count)

            if self.show_point_normals and buffer.normal_count > 0:
                glBindVertexArray(buffer.normal_vao)
                glUniform3f(self.override_loc, *colors_scheme['point_cloud_normals'])
                glDrawArrays(GL_LINES, 0, buffer.normal_count)

    def run(self):
        while not glfw.window_should_close(self.window):
            self.handle_input()
            
            colors_scheme = self.get_color_scheme()
            glClearColor(*colors_scheme['background'])
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            if self.point_buffers:
                self.render_pc()

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

if __name__ == "__main__":
    app = PointViewer()
    app.run()