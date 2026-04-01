# Shader Source Code
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 mvp;
uniform float pointSize;
out vec3 vColor;

void main() {
    gl_Position = mvp * vec4(aPos, 1.0);
    gl_PointSize = pointSize;
    vColor = aColor;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;

uniform vec3 overrideColor;
uniform bool useOverride;

void main() {
    if (useOverride) {
        FragColor = vec4(overrideColor, 1.0);
    } else {
        FragColor = vec4(vColor, 1.0);
    }
}
"""

# Window Settings
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
WINDOW_TITLE = "Mesh Viewer | O: Open File"
WINDOW_ASPECT_RATIO = WINDOW_WIDTH / WINDOW_HEIGHT

# Rendering Modes
MODE_SOLID = 0
MODE_WIREFRAME = 1
MODE_BOTH = 2

# Color Themes
THEME_DARK = 0
THEME_LIGHT = 1

# Color Schemes
COLOR_SCHEME_DARK = {
    'background': (0.05, 0.05, 0.1, 1.0),
    'mesh': (0.7, 0.7, 0.7),  # Light gray
    'wireframe': (0.5, 0.5, 0.5),  # Medium gray
    'wireframe_highlight': (1.0, 1.0, 1.0),  # White
    'intersected': (1.0, 0.5, 0.0),  # Orange
    'face_normals': (0.2, 0.8, 0.2),  # Green
    'vertex_normals': (0.2, 0.6, 1.0),  # Blue
    'point_cloud': (1.0, 1.0, 0.0),  # Yellow
    'point_cloud_normals': (1.0, 0.0, 1.0),  # Magenta
    'nonmanifold_edges': (1.0, 0.0, 0.0),  # Red
    'internal_edges': (0.6, 0.6, 0.6),  # Light gray
    'boundary_edges': (0.2, 0.6, 1.0),  # Blue
    'nonmanifold_vertices': (0.0, 1.0, 1.0)  # Cyan
}

COLOR_SCHEME_LIGHT = {
    'background': (0.95, 0.95, 0.95, 1.0),  # Light gray/white
    'mesh': (0.3, 0.3, 0.3),  # Dark gray
    'wireframe': (0.5, 0.5, 0.5),  # Medium gray
    'wireframe_highlight': (0.0, 0.0, 0.0),  # Black
    'intersected': (1.0, 0.5, 0.0),  # Orange
    'face_normals': (0.0, 0.6, 0.0),  # Dark green
    'vertex_normals': (0.0, 0.4, 0.8),  # Dark blue
    'point_cloud': (0.8, 0.8, 0.0),  # Dark yellow
    'point_cloud_normals': (1.0, 0.0, 1.0),  # Magenta
    'nonmanifold_edges': (0.8, 0.0, 0.0),  # Dark red
    'internal_edges': (0.3, 0.3, 0.3),  # Even lighter gray
    'boundary_edges': (0.0, 0.4, 0.8),  # Dark blue
    'nonmanifold_vertices': (0.0, 0.8, 0.8)  # Dark cyan
}

# Default Initial Values
DEFAULT_MODE = MODE_SOLID
DEFAULT_SHOW_INTERSECTED = False
DEFAULT_SHOW_FACE_NORMALS = False
DEFAULT_SHOW_VERTEX_NORMALS = False
DEFAULT_SHOW_POINT_CLOUD = False
DEFAULT_SHOW_POINT_CLOUD_NORMALS = False
DEFAULT_SHOW_NONMANIFOLD_EDGES = False
DEFAULT_SHOW_NONMANIFOLD_VERTICES = False
DEFAULT_COLOR_THEME = THEME_DARK

# Camera Defaults
DEFAULT_CAMERA_ROTATING = True
DEFAULT_CAMERA_ANGLE = 0.0
DEFAULT_CAMERA_VERTICAL_ANGLE = 0.0
DEFAULT_CAMERA_DISTANCE = 3.5
DEFAULT_CAMERA_HEIGHT = 1.0
DEFAULT_CAMERA_ROTATION_SPEED = 0.3  # Auto rotation speed (radians per second)
DEFAULT_CAMERA_MANUAL_SPEED = 0.1  # Manual rotation speed (radians per second)
DEFAULT_CAMERA_HEIGHT_SPEED = 1.5  # Units per second

# Object Defaults
DEFAULT_OBJECT_ROTATION_X = 0.0
DEFAULT_OBJECT_ROTATION_Y = 0.0
DEFAULT_OBJECT_ROTATION_Z = 0.0
DEFAULT_OBJECT_ROTATION_SPEED = 0.2  # Radians per second
DEFAULT_OBJECT_SCALE = 1.0
DEFAULT_OBJECT_SCALE_SPEED = 0.2  # Units per second

# Mesh Visualization
NORMAL_LENGTH_FACTOR = 0.02  # Factor of mesh diagonal
NORMAL_LENGTH_MIN = 0.01  # Minimum normal length
POINT_CLOUD_SAMPLE_COUNT = 8192
MESH_LAYOUT_PADDING = 0.2  # Extra spacing between meshes in grid layout
MESH_LAYOUT_PADDING_STEP = 0.1
MESH_LAYOUT_PADDING_MIN = 0.0
MESH_LAYOUT_PADDING_MAX = 5.0

# Point Cloud Visualization
PC_LAYOUT_PADDING = 0.2  # Extra spacing between point clouds in grid layout
PC_LAYOUT_PADDING_STEP = 0.1
PC_LAYOUT_PADDING_MIN = 0.0
PC_LAYOUT_PADDING_MAX = 5.0

# Camera Projection
CAMERA_FOV = 45.0  # Field of view in degrees
CAMERA_NEAR_PLANE = 0.1
CAMERA_FAR_PLANE = 100.0

# Camera Limits
CAMERA_HEIGHT_MIN = -10.0
CAMERA_HEIGHT_MAX = 10.0

# Object Scale Limits
OBJECT_SCALE_MIN = 0.1
OBJECT_SCALE_MAX = 10.0

# OpenGL Settings
VERTEX_STRIDE = 12  # Size in bytes: Pos(3*4)
COLOR_OFFSET = 12  # Deprecated: color is now controlled by uniforms
POLYGON_OFFSET_FACTOR = 1.0
POLYGON_OFFSET_UNITS = 1.0

# Point Size Settings
NONMANIFOLD_VERTEX_POINT_SIZE = 10.0  # Size of non-manifold vertex points
POINT_CLOUD_POINT_SIZE = 3.0  # Size of point cloud points

# Input/Timing
ASSUMED_FPS = 60.0
DELTA_TIME = 1.0 / ASSUMED_FPS

# File Dialog Settings
MESH_FILE_TYPES = [
    ("Mesh Files", "*.obj *.stl *.ply *.glb *.off"),
    ("All Files", "*.*")
]

POINT_CLOUD_FILE_TYPES = [
    ("Point Cloud Files", "*.ply *.xyz *.txt"),
    ("All Files", "*.*")
]

SCREENSHOT_FILE_TYPES = [
    ("PNG", "*.png"),
    ("JPEG", "*.jpg"),
    ("PDF", "*.pdf"),
    ("All Files", "*.*")
]

SCREENSHOT_DEFAULT_EXTENSION = ".png"

# Dialog Titles
DIALOG_TITLE_SELECT_MESH = "Select Mesh File"
DIALOG_TITLE_SAVE_SCREENSHOT = "Save Screenshot"
DIALOG_TITLE_SELECT_POINT_CLOUD = "Select Point Cloud File"

# Mesh Analysis Constants
COPLANAR_TOLERANCE = 1e-8  # Tolerance for coplanar vertex detection
NORMALIZE_BOUND = 0.95  # Default bound for vertex normalization
MANIFOLD_EDGE_COUNT = 2  # Expected edge count for manifold meshes
DUPLICATE_VERTICES_DECIMALS = 10  # Number of decimals for duplicate vertex detection
CHECK_GEOMETRY_SUGGESTION_PROMPT = "Turn on geometry checking for more detailed analysis (may increase processing time)."
CHECK_COMPONENTS_SUGGESTION_PROMPT = "Turn on connected component checking for more detailed analysis (may increase processing time)."
CHECK_TOPOLOGY_SUGGESTION_PROMPT = "Turn on topology checking for more detailed analysis (may increase processing time)."
CHECK_INTERSECTION_SUGGESTION_PROMPT = None # "Turn on intersection checking for more detailed analysis (may increase processing time)."
CHECK_MANIFOLD_VERTICES_SUGGESTION_PROMPT = None # "Turn on non-manifold vertex checking for more detailed analysis (may increase processing time)."

# Mesh Info Formatting Constants
FORMAT_LABEL_WIDTH = 40  # Width for label formatting in mesh info output
FORMAT_PRECISION_FLOAT = 6  # Decimal places for general float formatting
FORMAT_PRECISION_COORD = 3  # Decimal places for coordinate formatting
DEFAULT_CHECK_COMPONENTS = False
DEFAULT_CHECK_INTERSECTION = False
DEFAULT_CHECK_NONMANIFOLD_VERTICES = False
DEFAULT_CHECK_GEOMETRY = False
DEFAULT_CHECK_TOPOLOGY = False
DEFAULT_MAX_CONTACTS = 100000
DEFAULT_VERBOSE = False