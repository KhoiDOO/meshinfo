# Mesh Analysis Constants
COPLANAR_TOLERANCE = 1e-8  # Tolerance for coplanar vertex detection
MANIFOLD_EDGE_COUNT = 2  # Expected edge count for manifold meshes
DUPLICATE_VERTICES_DECIMALS = 10  # Number of decimals for duplicate vertex detection
CHECK_GEOMETRY_SUGGESTION_PROMPT = "Turn on geometry checking for more detailed analysis (may increase processing time)."
CHECK_COMPONENTS_SUGGESTION_PROMPT = "Turn on connected component checking for more detailed analysis (may increase processing time)."
CHECK_TOPOLOGY_SUGGESTION_PROMPT = "Turn on topology checking for more detailed analysis (may increase processing time)."
CHECK_INTERSECTION_SUGGESTION_PROMPT = None
CHECK_MANIFOLD_VERTICES_SUGGESTION_PROMPT = None

# Mesh Info Formatting Constants
FORMAT_LABEL_WIDTH = 40  # Width for label formatting in mesh info output
FORMAT_PRECISION_FLOAT = 6  # Decimal places for general float formatting
FORMAT_PRECISION_COORD = 3  # Decimal places for coordinate formatting

# Default Initial Values (Analysis)
DEFAULT_CHECK_COMPONENTS = False
DEFAULT_CHECK_INTERSECTION = False
DEFAULT_CHECK_NONMANIFOLD_VERTICES = False
DEFAULT_CHECK_GEOMETRY = False
DEFAULT_CHECK_TOPOLOGY = False
DEFAULT_VERBOSE = False
