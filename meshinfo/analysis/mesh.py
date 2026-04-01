import trimesh
import numpy as np
import meshlib.mrmeshpy as mrmeshpy
import meshlib.mrmeshnumpy as mrmeshnumpy
import warnings
from colorama import Fore, Style, init
from . import format_value, format_bool

from ..constants import *

from trimesh.triangles import mass_properties, MassProperties

# Initialize colorama for Windows compatibility
init(autoreset=True)

def is_manifold(edge_counts) -> bool:
    is_manifold = np.all(edge_counts == MANIFOLD_EDGE_COUNT).item()
    return is_manifold

def get_intersected_tria_ids(mesh: trimesh.Trimesh, max_num_contacts: int = 1000) -> list[int]:
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

    return list(intersected_ids)

def get_nonmanifold_vertices(vertices, faces, edges_unique, edges_counts, face_adjacency) -> np.ndarray:
    nonmanifold_vertices = set()
    
    # First, collect vertices on non-manifold edges
    nonmanifold_edge_mask = edges_counts != 2  # Edges shared by more or less than 2 faces
    if np.any(nonmanifold_edge_mask):
        nonmanifold_edge_vertices = edges_unique[nonmanifold_edge_mask].flatten()
        nonmanifold_vertices.update(nonmanifold_edge_vertices)
    else:
        return np.array([], dtype=np.int32)
    
    # Preprocess face adjacency into a more efficient lookup structure
    # Build a dictionary: face_id -> list of adjacent face_ids
    face_adjacency_dict: dict[int, list[int]] = {}
    for face_pair in face_adjacency:
        f1, f2 = face_pair[0], face_pair[1]
        if f1 not in face_adjacency_dict:
            face_adjacency_dict[f1] = []
        if f2 not in face_adjacency_dict:
            face_adjacency_dict[f2] = []
        face_adjacency_dict[f1].append(f2)
        face_adjacency_dict[f2].append(f1)
    
    # For each vertex, check if its adjacent faces form a single connected component
    for vertex_idx in range(len(vertices)):
        # Get all faces adjacent to this vertex
        adjacent_faces = np.where(np.any(faces == vertex_idx, axis=1))[0]
        
        if len(adjacent_faces) < 2:
            continue
        
        # Check connectivity using preprocessed face_adjacency_dict
        adjacent_faces_set = set(adjacent_faces)
        visited = set()
        stack = [adjacent_faces[0]]
        visited.add(adjacent_faces[0])
        
        while stack:
            current_face = stack.pop()
            # Get neighbors from the preprocessed dictionary
            if current_face in face_adjacency_dict:
                for neighbor_face in face_adjacency_dict[current_face]:
                    # Only consider neighbors that are also adjacent to this vertex
                    if neighbor_face in adjacent_faces_set and neighbor_face not in visited:
                        visited.add(neighbor_face)
                        stack.append(neighbor_face)
        
        # If not all faces are connected, vertex is non-manifold
        if len(visited) != len(adjacent_faces):
            nonmanifold_vertices.add(vertex_idx)
    
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

def get_volume_center_mass_density(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asanyarray(triangles, dtype=np.float64)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        _mass_properties: MassProperties = mass_properties(triangles, skip_inertia=True)

    volume = _mass_properties.volume
    center_mass = _mass_properties.center_mass if volume != 0 else None

    return volume, center_mass


class MeshInfo:
    def __init__(
        self, 
        mesh: trimesh.Trimesh, 
        name: str = "Mesh",
        check_components: bool = DEFAULT_CHECK_COMPONENTS,
        check_intersection: bool = DEFAULT_CHECK_INTERSECTION,
        check_nonmanifold_vertices: bool = DEFAULT_CHECK_NONMANIFOLD_VERTICES,
        check_geometry: bool = DEFAULT_CHECK_GEOMETRY,
        check_topology: bool = DEFAULT_CHECK_TOPOLOGY,
        verbose: bool = DEFAULT_VERBOSE
    ):
        self.mesh = mesh
        self.name = name
        self.verbose = verbose

        # General Properties
        if verbose: print(f"{Fore.YELLOW}Computing general properties...{Style.RESET_ALL}")
        self.euler = mesh.euler_number
        self.genus = 1 - self.euler / 2
        self.edges_unique: np.ndarray
        self.edges_counts: np.ndarray
        self.edges_unique, self.edges_counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)

        # Geometric Properties
        self.check_geometry = check_geometry
        if check_geometry:
            if verbose: print(f"{Fore.YELLOW}Computing geometric properties...{Style.RESET_ALL}")
            self.volume, self.center_mass = get_volume_center_mass_density(mesh.triangles)
            self.area = mesh.area
            self.bounds = mesh.bounds
            self.extents = np.ptp(self.bounds, axis=0)
        else:
            if verbose: print(f"{Fore.YELLOW}Skipping geometric properties. {CHECK_GEOMETRY_SUGGESTION_PROMPT}{Style.RESET_ALL}")
        
        # Connected Components
        self.checked_components = check_components
        self.check_topology = check_topology
        if self.checked_components:
            if verbose: print(f"{Fore.YELLOW}Computing connected components...{Style.RESET_ALL}")
            self.body_count = mesh.body_count if check_components else CHECK_COMPONENTS_SUGGESTION_PROMPT
            self.non_watertight_components = mesh.split(only_watertight=False) if check_components else []
            self.watertight_components = mesh.split(only_watertight=True) if check_components else []
            self.wt_css_f_num = [len(c.faces) for c in self.watertight_components]
            self.nwt_css_f_num = [len(c.faces) for c in self.non_watertight_components]
            self.wt_css_v_num = [len(c.vertices) for c in self.watertight_components]
            self.nwt_css_v_num = [len(c.vertices) for c in self.non_watertight_components]
        else:
            if verbose: print(f"{Fore.YELLOW}Skipping connected components. {CHECK_COMPONENTS_SUGGESTION_PROMPT}{Style.RESET_ALL}")

        # Vertex Properties
        if verbose: print(f"{Fore.YELLOW}Computing vertex properties...{Style.RESET_ALL}")
        self.vertex_defects: np.ndarray = mesh.vertex_defects
        self.vertex_degree: np.ndarray = mesh.vertex_degree

        unique_vertices = np.unique(np.round(mesh.vertices, decimals=DUPLICATE_VERTICES_DECIMALS), axis=0)
        self.num_duplicate_vertices = len(mesh.vertices) - len(unique_vertices)

        self.checked_nonmanifold_vertices = check_nonmanifold_vertices
        self.nonmanifold_vertices = get_nonmanifold_vertices(
            mesh.vertices, mesh.faces, self.edges_unique, self.edges_counts, mesh.face_adjacency
        ) if check_nonmanifold_vertices else []
        self.num_nonmanifold_vertices = len(self.nonmanifold_vertices) \
            if check_nonmanifold_vertices else CHECK_MANIFOLD_VERTICES_SUGGESTION_PROMPT

        # Edge Properties
        if verbose: print(f"{Fore.YELLOW}Computing edge properties...{Style.RESET_ALL}")
        self.vertex_connectivity = np.bincount(self.edges_unique.flatten(), minlength=len(mesh.vertices))
        self.edges_unique_length: np.ndarray
        self.edges_unique_length = self.mesh.edges_unique_length
        
        self.nonmanifold_edge_mask = self.edges_counts > 2  # Edges shared by more than 2 faces
        self.nonmanifold_edges = self.edges_unique[self.nonmanifold_edge_mask]
        self.num_nonmanifold_edges = np.sum(self.nonmanifold_edge_mask).item()
        
        # Identify internal and boundary edges for visualization
        self.internal_edges = self.edges_unique[self.edges_counts == 2]
        self.boundary_edges = self.edges_unique[self.edges_counts == 1]

        # Intersection and Manifold Checks
        if verbose: print(f"{Fore.YELLOW}Computing intersection and manifold properties...{Style.RESET_ALL}")
        self.checked_intersection = check_intersection
        self.intersected_face_ids = get_intersected_tria_ids(mesh) \
            if check_intersection else []
        self.num_intersected_faces = len(self.intersected_face_ids) \
            if check_intersection else CHECK_INTERSECTION_SUGGESTION_PROMPT
        self.is_intersecting = self.num_intersected_faces > 0 \
            if check_intersection else CHECK_INTERSECTION_SUGGESTION_PROMPT
        self.is_manifold_ignore_intersection = is_manifold(self.edges_counts)
        self.is_manifold = self.is_manifold_ignore_intersection and not self.is_intersecting \
            if check_intersection else CHECK_INTERSECTION_SUGGESTION_PROMPT
        
        # Face Properties
        if verbose: print(f"{Fore.YELLOW}Computing face properties...{Style.RESET_ALL}")
        self.nondegenerate_faces_mask = mesh.nondegenerate_faces()
        self.num_degenerate_faces = np.sum(~self.nondegenerate_faces_mask).item()
        self.num_nondegenerate_faces = np.sum(self.nondegenerate_faces_mask).item()
        self.face_angles: np.ndarray = mesh.face_angles
        self.face_areas: np.ndarray = mesh.area_faces
        self.face_adjacency_angles: np.ndarray = mesh.face_adjacency_angles
        self.num_dup_faces = get_num_dup_faces(mesh)
        
        if verbose: print(f"{Fore.YELLOW}Compiling statistics and properties...{Style.RESET_ALL}")
        self.stats = {
            "#vertices": len(mesh.vertices),
            "#faces": len(mesh.faces),
            "#edges": len(mesh.edges_unique),
            "euler": self.euler,
            "genus": self.genus,
        }

        self.properties = {
            "is_watertight": mesh.is_watertight,
            "is_empty": mesh.is_empty,
            "is_winding_consistent": mesh.is_winding_consistent,
            "is_convex": mesh.is_convex,
            "is_manifold[ignore intersection]": self.is_manifold_ignore_intersection,
            "is_manifold": self.is_manifold,
            "mutable": mesh.mutable,
            "is_intersecting": self.is_intersecting,
            "symmetry": mesh.symmetry
        } if check_topology else {
            "INFO": CHECK_TOPOLOGY_SUGGESTION_PROMPT
        }

        self.analysis = {
            "area": self.area,
            "volume": self.volume,
            "sphericity": get_sphericity(self.volume, self.area),
            "bounds": self.bounds,
            "center_mass": self.center_mass,
            "centroid": mesh.centroid,
            "extents": self.extents,
        } if check_geometry else {
            "INFO": CHECK_GEOMETRY_SUGGESTION_PROMPT
        }

        self.vertices_info = {
            "#coplanar_vertices": np.sum(np.abs(self.vertex_defects) < COPLANAR_TOLERANCE).item(),
            "#convex_vertices": np.sum(self.vertex_defects > 0).item(),
            "#concave_vertices": np.sum(self.vertex_defects < 0).item(),
            "#duplicate_vertices": self.num_duplicate_vertices,
            "min_v_degree": int(self.vertex_degree.min()),
            "max_v_degree": int(self.vertex_degree.max()),
        }

        self.edges_info = {
            "#internal_edges": np.sum(self.edges_counts == 2),
            "#boundary_edges": np.sum(self.edges_counts == 1),
            "#nonmanifold_edges": self.num_nonmanifold_edges,
            "#nonmanifold_vertices": self.num_nonmanifold_vertices,
            "min_connectivity": int(self.vertex_connectivity.min()),
            "max_connectivity": int(self.vertex_connectivity.max()),
            "avg_connectivity": float(self.vertex_connectivity.mean()),
            "min_edge_length[mel]": float(self.edges_unique_length.min()),
            "max_edge_length[mal]": float(self.edges_unique_length.max()),
        }
        self.edges_info["aspect_ratio[ar][mal/mel]"] = self.edges_info["max_edge_length[mal]"] / self.edges_info["min_edge_length[mel]"] if self.edges_info["min_edge_length[mel]"] > 0 else float('inf')

        self.faces_info = {
            "#intersected_faces": self.num_intersected_faces,
            "#degenerate_faces": self.num_degenerate_faces,
            "#non_degenerate_faces": self.num_nondegenerate_faces,
            "min_f_angle[rad]": float(self.face_angles.min()),
            "max_f_angle[rad]": float(self.face_angles.max()),
            "min_f_angle[deg]": float(np.degrees(self.face_angles.min())),
            "max_f_angle[deg]": float(np.degrees(self.face_angles.max())),
            "min_f_area": float(self.face_areas.min()),
            "max_f_area": float(self.face_areas.max()),
            "min_dihedral_angle[deg]": float(np.degrees(np.min(self.face_adjacency_angles))) if len(self.face_adjacency_angles) > 0 else None,
            "max_dihedral_angle[deg]": float(np.degrees(np.max(self.face_adjacency_angles))) if len(self.face_adjacency_angles) > 0 else None,
            "num_dup_faces": self.num_dup_faces,
        }

        self.ccs_info = {
            "#ccs": self.body_count,
            "#ccs[split][wt=True]": len(self.watertight_components),
            "#ccs[split][wt=False]": len(self.non_watertight_components),
            "ccs_max_f_wt" : max(self.wt_css_f_num) if len(self.watertight_components) > 0 else self.stats['#faces'],
            "ccs_max_f_non_wt" : max(self.nwt_css_f_num) if len(self.non_watertight_components) > 0 else self.stats['#faces'],
            "ccs_min_f_wt" : min(self.wt_css_f_num) if len(self.watertight_components) > 0 else self.stats['#faces'],
            "ccs_min_f_non_wt" : min(self.nwt_css_f_num) if len(self.non_watertight_components) > 0 else self.stats['#faces'],
            "css_max_v_wt" : max(self.wt_css_v_num) if len(self.watertight_components) > 0 else self.stats['#vertices'],
            "css_max_v_non_wt" : max(self.nwt_css_v_num) if len(self.non_watertight_components) > 0 else self.stats['#vertices'],
            "css_min_v_wt" : min(self.wt_css_v_num) if len(self.watertight_components) > 0 else self.stats['#vertices'],
            "css_min_v_non_wt" : min(self.nwt_css_v_num) if len(self.non_watertight_components) > 0 else self.stats['#vertices'],
        } if self.checked_components else {
            "INFO": CHECK_COMPONENTS_SUGGESTION_PROMPT
        }
    
    def __str__(self):
        info_str = f"{Fore.CYAN}{Style.BRIGHT}╔═══ Mesh Information [{self.name}] ═══╗{Style.RESET_ALL}\n"
        
        # Statistics
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Statistics:{Style.RESET_ALL}\n"
        info_str += f"  {Fore.CYAN}{'#vertices / #faces / #edges':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL}"
        info_str += f" {format_value(self.stats['#vertices'])} / {format_value(self.stats['#faces'])} / {format_value(self.stats['#edges'])}\n"
        
        for key, value in self.stats.items():
            if key not in ['#vertices', '#faces', '#edges']:
                formatted_value = format_value(value)
                info_str += f"  {Fore.CYAN}{key:.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {formatted_value}\n"
        
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Properties:{Style.RESET_ALL}\n"
        if self.check_topology:
            # Row 1: Topological properties
            info_str += f"  {Fore.CYAN}watertight:{Style.RESET_ALL} {format_bool(self.properties['is_watertight'])}  "
            info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}manifold (ignore intersection):{Style.RESET_ALL} {format_bool(self.properties['is_manifold[ignore intersection]'])}  {Fore.CYAN}manifold:{Style.RESET_ALL} {format_bool(self.properties['is_manifold'])}  "
            info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}winding_consistent:{Style.RESET_ALL} {format_bool(self.properties['is_winding_consistent'])}\n"
            
            # Row 2: Geometric and state properties
            info_str += f"  {Fore.CYAN}convex:{Style.RESET_ALL} {format_bool(self.properties['is_convex'])}  "
            info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}empty:{Style.RESET_ALL} {format_bool(self.properties['is_empty'])}  "
            info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}intersecting:{Style.RESET_ALL} {format_bool(self.properties['is_intersecting'])}  "
            info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}mutable:{Style.RESET_ALL} {format_bool(self.properties['mutable'])}\n"
            info_str += f"  {Fore.CYAN}symmetry:{Style.RESET_ALL} {format_value(self.properties['symmetry'])}\n"
        else:
            info_str += f"  {Fore.CYAN}{'INFO':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {CHECK_TOPOLOGY_SUGGESTION_PROMPT}\n"
        
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Analysis:{Style.RESET_ALL}\n"
        if self.check_geometry:
            for key, value in self.analysis.items():
                if key == "bounds":
                    key = "bounds[unnormalized]"
                    value_str = f"{Fore.YELLOW}[[{value[0][0]:.{FORMAT_PRECISION_COORD}f}, {value[0][1]:.{FORMAT_PRECISION_COORD}f}, {value[0][2]:.{FORMAT_PRECISION_COORD}f}], "
                    value_str += f"[{value[1][0]:.{FORMAT_PRECISION_COORD}f}, {value[1][1]:.{FORMAT_PRECISION_COORD}f}, {value[1][2]:.{FORMAT_PRECISION_COORD}f}]]{Style.RESET_ALL}"
                    info_str += f"  {Fore.CYAN}{key:.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {value_str}\n"
                elif key == "extents":
                    value_str = f"{Fore.YELLOW}[l = {value[0]:.{FORMAT_PRECISION_COORD}f}, w = {value[1]:.{FORMAT_PRECISION_COORD}f}, h = {value[2]:.{FORMAT_PRECISION_COORD}f}]{Style.RESET_ALL}"
                    info_str += f"  {Fore.CYAN}{key:.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {value_str}\n"
                else:
                    formatted_value = format_value(value)
                    info_str += f"  {Fore.CYAN}{key:.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {formatted_value}\n"
        else:
            info_str += f"  {Fore.CYAN}{'INFO':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {CHECK_GEOMETRY_SUGGESTION_PROMPT}\n"
        
        # Vertices Info - group related items
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Vertices Info:{Style.RESET_ALL}\n"
        info_str += f"  {Fore.CYAN}{'#coplanar / #convex / #concave':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.vertices_info['#coplanar_vertices'])} / "
        info_str += f"{format_value(self.vertices_info['#convex_vertices'])} / "
        info_str += f"{format_value(self.vertices_info['#concave_vertices'])}\n"
        
        info_str += f"  {Fore.CYAN}{'#duplicate_vertices':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.vertices_info['#duplicate_vertices'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max vertex_degree':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.vertices_info['min_v_degree'])} / "
        info_str += f"{format_value(self.vertices_info['max_v_degree'])}\n"
        
        # Edges Info - group min/max pairs
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Edges Info:{Style.RESET_ALL}\n"
        info_str += f"  {Fore.CYAN}{'#internal / #boundary / #nonmanifold':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.edges_info['#internal_edges'])} / "
        info_str += f"{format_value(self.edges_info['#boundary_edges'])} / "
        info_str += f"{format_value(self.edges_info['#nonmanifold_edges'])}\n"
        
        info_str += f"  {Fore.CYAN}{'#nonmanifold_vertices':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.edges_info['#nonmanifold_vertices'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max / avg connectivity':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.edges_info['min_connectivity'])} / "
        info_str += f"{format_value(self.edges_info['max_connectivity'])} / "
        info_str += f"{format_value(self.edges_info['avg_connectivity'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max edge_length[mel/mal]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.edges_info['min_edge_length[mel]'])} / "
        info_str += f"{format_value(self.edges_info['max_edge_length[mal]'])}\n"
        
        info_str += f"  {Fore.CYAN}{'aspect_ratio[ar][mal/mel]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.edges_info['aspect_ratio[ar][mal/mel]'])}\n"

        # Faces Info - group min/max pairs
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Faces Info:{Style.RESET_ALL}\n"
        info_str += f"  {Fore.CYAN}{'#intersected_faces':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.faces_info['#intersected_faces'])}\n"
        info_str += f"  {Fore.CYAN}{'#degenerate / #non_degenerate':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['#degenerate_faces'])} / "
        info_str += f"{format_value(self.faces_info['#non_degenerate_faces'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max face_angle[rad]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['min_f_angle[rad]'])} / "
        info_str += f"{format_value(self.faces_info['max_f_angle[rad]'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max face_angle[deg]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['min_f_angle[deg]'])} / "
        info_str += f"{format_value(self.faces_info['max_f_angle[deg]'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max face_area':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['min_f_area'])} / "
        info_str += f"{format_value(self.faces_info['max_f_area'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max dihedral_angle[deg]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['min_dihedral_angle[deg]'])} / "
        info_str += f"{format_value(self.faces_info['max_dihedral_angle[deg]'])}\n"

        info_str += f"  {Fore.CYAN}{'num_dup_faces':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.faces_info['num_dup_faces'])}\n"

        # Connected Components Info
        if self.checked_components:
            info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Connected Components Info:{Style.RESET_ALL}\n"
            info_str += f"  {Fore.CYAN}{'#ccs':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.ccs_info['#ccs'])}\n"
            info_str += f"  {Fore.CYAN}{'#ccs[split]: [wt=True]/[wt=False]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.ccs_info['#ccs[split][wt=True]'])} / {format_value(self.ccs_info['#ccs[split][wt=False]'])}\n"
            info_str += f"  {Fore.CYAN}{'ccs_max_f_wt / ccs_min_f_wt':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.ccs_info['ccs_max_f_wt'])} / {format_value(self.ccs_info['ccs_min_f_wt'])}\n"
            info_str += f"  {Fore.CYAN}{'ccs_max_f_non_wt / ccs_min_f_non_wt':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.ccs_info['ccs_max_f_non_wt'])} / {format_value(self.ccs_info['ccs_min_f_non_wt'])}\n"
            info_str += f"  {Fore.CYAN}{'css_max_v_wt / css_min_v_wt':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.ccs_info['css_max_v_wt'])} / {format_value(self.ccs_info['css_min_v_wt'])}\n"
            info_str += f"  {Fore.CYAN}{'css_max_v_non_wt / css_min_v_non_wt':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.ccs_info['css_max_v_non_wt'])} / {format_value(self.ccs_info['css_min_v_non_wt'])}\n"
        else:
            info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Connected Components Info:{Style.RESET_ALL}\n"
            info_str += f"  {Fore.CYAN}{'INFO':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {CHECK_COMPONENTS_SUGGESTION_PROMPT}\n"

        info_str += f"\n{Fore.CYAN}{Style.BRIGHT}╚═══════════════════════╝{Style.RESET_ALL}"
        return info_str

    def to_dict(self, np2list=True, nested=False) -> dict:

        info_dict = {
            "stats": self.stats,
            "properties": self.properties,
            "analysis": self.analysis,
            "vertices_info": self.vertices_info,
            "edges_info": self.edges_info,
            "faces_info": self.faces_info,
            "ccs_info": self.ccs_info
        } if nested else {
            **self.stats,
            **self.properties,
            **self.analysis,
            **self.vertices_info,
            **self.edges_info,
            **self.faces_info,
            **self.ccs_info
        }

        if np2list:
            if nested:
                for section_key, section_dict in info_dict.items():
                    for key, value in section_dict.items():
                        if isinstance(value, np.ndarray):
                            if value.ndim == 0:
                                info_dict[section_key][key] = value.item()
                            info_dict[section_key][key] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            info_dict[section_key][key] = value.item()
            else:
                for key, value in info_dict.items():
                    if isinstance(value, np.ndarray):
                        if value.ndim == 0:
                            info_dict[key] = value.item()
                        info_dict[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        info_dict[key] = value.item()

        return info_dict
