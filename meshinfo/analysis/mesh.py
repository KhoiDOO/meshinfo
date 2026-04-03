import trimesh
import numpy as np
import meshlib.mrmeshpy as mrmeshpy
import meshlib.mrmeshnumpy as mrmeshnumpy
import networkx as nx
from colorama import Fore, Style, init
from . import format_value, format_bool

from ..constants import *

# Initialize colorama for Windows compatibility
init(autoreset=True)

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
    for vertex_idx, face_indices in enumerate(mesh.vertex_faces):
        # Filter out the padding (-1)
        face_indices = face_indices[face_indices != -1]
        
        if vertex_idx in nonmanifold_vertices or len(face_indices) < 2:
            continue
        
        # Build a local adjacency graph for faces sharing this vertex
        local_faces = mesh.faces[face_indices]
        G = nx.Graph()
        G.add_nodes_from(range(len(face_indices)))
        
        for i in range(len(face_indices)):
            for j in range(i + 1, len(face_indices)):
                # Share an edge if they share 2 vertices (one is vertex_idx)
                shared_v = np.intersect1d(local_faces[i], local_faces[j])
                if len(shared_v) >= 2:
                    G.add_edge(i, j)
        
        if not nx.is_connected(G):
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
        self.checked_components = check_components
        self.check_geometry = check_geometry
        self.check_topology = check_topology
        self.checked_intersection = check_intersection
        self.checked_nonmanifold_vertices = check_nonmanifold_vertices

        # General Properties
        if verbose: print(f"{Fore.YELLOW}Computing general properties...{Style.RESET_ALL}")
        self.num_vertices = len(mesh.vertices)
        self.num_faces = len(mesh.faces)
        self.euler = mesh.euler_number
        self.genus = 1 - self.euler / 2
        self.edges_unique, self.edges_counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        self.num_edges = len(self.edges_unique)

        # Topological Properties
        if check_topology:
            self.is_watertight = mesh.is_watertight
            self.is_empty = mesh.is_empty
            self.is_winding_consistent = mesh.is_winding_consistent
            self.is_convex = mesh.is_convex
            self.is_mutable = mesh.mutable
            self.symmetry = mesh.symmetry
        else:
            self.is_watertight = self.is_empty = self.is_winding_consistent = self.is_convex = self.is_mutable = self.symmetry = None

        # Geometric Properties
        if check_geometry:
            if verbose: print(f"{Fore.YELLOW}Computing geometric properties...{Style.RESET_ALL}")
            self.volume = mesh.volume
            self.area = mesh.area
            self.sphericity = get_sphericity(self.volume, self.area)
            self.bounds = mesh.bounds
            self.bounds_list = self.bounds.tolist() if self.bounds is not None else None
            self.center_mass = mesh.center_mass if abs(self.volume) > 1e-12 else None
            self.center_mass_list = self.center_mass.tolist() if self.center_mass is not None else None
            self.centroid_list = mesh.centroid.tolist()
            self.extents = np.ptp(self.bounds, axis=0)
            self.extents_list = self.extents.tolist() if self.extents is not None else None
        else:
            if verbose: print(f"{Fore.YELLOW}Skipping geometric properties. {CHECK_GEOMETRY_SUGGESTION_PROMPT}{Style.RESET_ALL}")
            self.volume = self.area = self.sphericity = self.bounds = self.bounds_list = self.center_mass = self.center_mass_list = self.centroid_list = self.extents = self.extents_list = None
        
        # Connected Components
        if self.checked_components:
            if verbose: print(f"{Fore.YELLOW}Computing connected components...{Style.RESET_ALL}")
            self.body_count = mesh.body_count
            self.non_watertight_components = mesh.split(only_watertight=False)
            self.watertight_components = mesh.split(only_watertight=True)
            self.wt_ccs_f_num = [len(c.faces) for c in self.watertight_components]
            self.nwt_ccs_f_num = [len(c.faces) for c in self.non_watertight_components]
            self.wt_ccs_v_num = [len(c.vertices) for c in self.watertight_components]
            self.nwt_ccs_v_num = [len(c.vertices) for c in self.non_watertight_components]
            
            self.num_ccs_wt = len(self.watertight_components)
            self.num_ccs_nwt = len(self.non_watertight_components)
            self.ccs_max_f_wt = max(self.wt_ccs_f_num) if self.num_ccs_wt > 0 else self.num_faces
            self.ccs_max_f_nwt = max(self.nwt_ccs_f_num) if self.num_ccs_nwt > 0 else self.num_faces
            self.ccs_min_f_wt = min(self.wt_ccs_f_num) if self.num_ccs_wt > 0 else self.num_faces
            self.ccs_min_f_nwt = min(self.nwt_ccs_f_num) if self.num_ccs_nwt > 0 else self.num_faces
            self.ccs_max_v_wt = max(self.wt_ccs_v_num) if self.num_ccs_wt > 0 else self.num_vertices
            self.ccs_max_v_nwt = max(self.nwt_ccs_v_num) if self.num_ccs_nwt > 0 else self.num_vertices
            self.ccs_min_v_wt = min(self.wt_ccs_v_num) if self.num_ccs_wt > 0 else self.num_vertices
            self.ccs_min_v_nwt = min(self.nwt_ccs_v_num) if self.num_ccs_nwt > 0 else self.num_vertices
        else:
            if verbose: print(f"{Fore.YELLOW}Skipping connected components. {CHECK_COMPONENTS_SUGGESTION_PROMPT}{Style.RESET_ALL}")
            self.body_count = self.num_ccs_wt = self.num_ccs_nwt = None
            self.ccs_max_f_wt = self.ccs_max_f_nwt = self.ccs_min_f_wt = self.ccs_min_f_nwt = None
            self.ccs_max_v_wt = self.ccs_max_v_nwt = self.ccs_min_v_wt = self.ccs_min_v_nwt = None

        # Vertex Properties
        if verbose: print(f"{Fore.YELLOW}Computing vertex properties...{Style.RESET_ALL}")
        self.vertex_defects: np.ndarray = mesh.vertex_defects
        self.vertex_degree: np.ndarray = mesh.vertex_degree
        self.num_coplanar_vertices = np.sum(np.abs(self.vertex_defects) < COPLANAR_TOLERANCE).item()
        self.num_convex_vertices = np.sum(self.vertex_defects > 0).item()
        self.num_concave_vertices = np.sum(self.vertex_defects < 0).item()
        self.min_v_degree = int(self.vertex_degree.min())
        self.max_v_degree = int(self.vertex_degree.max())

        unique_vertices = np.unique(np.round(mesh.vertices, decimals=DUPLICATE_VERTICES_DECIMALS), axis=0)
        self.num_duplicate_vertices = len(mesh.vertices) - len(unique_vertices)

        self.nonmanifold_vertices = get_nonmanifold_vertices(
            mesh, self.edges_unique, self.edges_counts
        ) if check_nonmanifold_vertices else []
        self.num_nonmanifold_vertices = len(self.nonmanifold_vertices) \
            if check_nonmanifold_vertices else CHECK_MANIFOLD_VERTICES_SUGGESTION_PROMPT

        # Edge Properties
        if verbose: print(f"{Fore.YELLOW}Computing edge properties...{Style.RESET_ALL}")
        self.vertex_connectivity = np.bincount(self.edges_unique.flatten(), minlength=len(mesh.vertices))
        self.min_connectivity = int(self.vertex_connectivity.min())
        self.max_connectivity = int(self.vertex_connectivity.max())
        self.avg_connectivity = float(self.vertex_connectivity.mean())

        self.edges_unique_length: np.ndarray = self.mesh.edges_unique_length
        self.min_edge_length = float(self.edges_unique_length.min())
        self.max_edge_length = float(self.edges_unique_length.max())
        self.edge_aspect_ratio = self.max_edge_length / self.min_edge_length if self.min_edge_length > 0 else float('inf')
        
        self.nonmanifold_edge_mask = self.edges_counts > 2  # Edges shared by more than 2 faces
        self.nonmanifold_edges = self.edges_unique[self.nonmanifold_edge_mask]
        self.num_nonmanifold_edges = np.sum(self.nonmanifold_edge_mask).item()
        
        # Identify internal and boundary edges for visualization
        self.internal_edges = self.edges_unique[self.edges_counts == 2]
        self.boundary_edges = self.edges_unique[self.edges_counts == 1]
        self.num_internal_edges = len(self.internal_edges)
        self.num_boundary_edges = len(self.boundary_edges)

        # Intersection and Manifold Checks
        if verbose: print(f"{Fore.YELLOW}Computing intersection and manifold properties...{Style.RESET_ALL}")
        if check_intersection:
            self.intersected_face_ids, self.num_intersected_pairs = get_intersected_tria_ids(mesh)
            self.num_intersected_faces = len(self.intersected_face_ids)
            self.is_intersecting = self.num_intersected_faces > 0
            self.intersected_faces_ratio_pct = (self.num_intersected_faces / self.num_faces * 100)
        else:
            self.intersected_face_ids = []
            self.num_intersected_pairs = self.num_intersected_faces = self.is_intersecting = self.intersected_faces_ratio_pct = CHECK_INTERSECTION_SUGGESTION_PROMPT

        self.is_manifold_ignore_intersection = np.all(self.edges_counts == MANIFOLD_EDGE_COUNT).item()
        self.is_manifold = self.is_manifold_ignore_intersection and not self.is_intersecting \
            if check_intersection else CHECK_INTERSECTION_SUGGESTION_PROMPT
        
        # Face Properties
        if verbose: print(f"{Fore.YELLOW}Computing face properties...{Style.RESET_ALL}")
        self.nondegenerate_faces_mask = mesh.nondegenerate_faces()
        self.num_degenerate_faces = np.sum(~self.nondegenerate_faces_mask).item()
        self.num_nondegenerate_faces = np.sum(self.nondegenerate_faces_mask).item()
        self.face_angles: np.ndarray = mesh.face_angles
        self.min_f_angle_rad = float(self.face_angles.min())
        self.max_f_angle_rad = float(self.face_angles.max())
        self.min_f_angle_deg = float(np.degrees(self.min_f_angle_rad))
        self.max_f_angle_deg = float(np.degrees(self.max_f_angle_rad))

        self.face_areas: np.ndarray = mesh.area_faces
        self.min_f_area = float(self.face_areas.min())
        self.max_f_area = float(self.face_areas.max())

        self.face_adjacency_angles: np.ndarray = mesh.face_adjacency_angles
        self.min_dihedral_angle_deg = float(np.degrees(np.min(self.face_adjacency_angles))) if len(self.face_adjacency_angles) > 0 else None
        self.max_dihedral_angle_deg = float(np.degrees(np.max(self.face_adjacency_angles))) if len(self.face_adjacency_angles) > 0 else None
        
        self.num_dup_faces = get_num_dup_faces(mesh)

        # Deciles for better analysis of distribution (10th, 20th, ..., 90th percentile)
        percentiles = np.arange(10, 100, 10)
        if len(self.face_angles) > 0:
            angles_p = np.percentile(np.degrees(self.face_angles.flatten()), percentiles)
            self.face_angles_deciles = {f"{p}%": float(val) for p, val in zip(percentiles, angles_p)}
        else:
            self.face_angles_deciles = {}

        if len(self.face_areas) > 0:
            areas_p = np.percentile(self.face_areas, percentiles)
            self.face_areas_deciles = {f"{p}%": float(val) for p, val in zip(percentiles, areas_p)}
        else:
            self.face_areas_deciles = {}
        
        if len(self.face_adjacency_angles) > 0:
            dihedral_p = np.percentile(np.degrees(self.face_adjacency_angles), percentiles)
            self.face_adjacency_angles_deciles = {f"{p}%": float(val) for p, val in zip(percentiles, dihedral_p)}
        else:
            self.face_adjacency_angles_deciles = {}
        
        if verbose: print(f"{Fore.YELLOW}Compiling statistics and properties...{Style.RESET_ALL}")
        self.stats = {
            "#vertices": self.num_vertices,
            "#faces": self.num_faces,
            "#edges": self.num_edges,
            "euler": self.euler,
            "genus": self.genus,
        }

        self.properties = {
            "is_watertight": self.is_watertight,
            "is_empty": self.is_empty,
            "is_winding_consistent": self.is_winding_consistent,
            "is_convex": self.is_convex,
            "is_manifold[ignore intersection]": self.is_manifold_ignore_intersection,
            "is_manifold": self.is_manifold,
            "mutable": self.is_mutable,
            "is_intersecting": self.is_intersecting,
            "symmetry": self.symmetry
        } if check_topology else {
            "INFO": CHECK_TOPOLOGY_SUGGESTION_PROMPT
        }

        self.analysis = {
            "area": self.area,
            "volume": self.volume,
            "sphericity": self.sphericity,
            "bounds": self.bounds_list,
            "center_mass": self.center_mass_list,
            "centroid": self.centroid_list,
            "extents": self.extents_list,
        } if check_geometry else {
            "INFO": CHECK_GEOMETRY_SUGGESTION_PROMPT
        }

        self.vertices_info = {
            "#coplanar_vertices": self.num_coplanar_vertices,
            "#convex_vertices": self.num_convex_vertices,
            "#concave_vertices": self.num_concave_vertices,
            "#duplicate_vertices": self.num_duplicate_vertices,
            "min_v_degree": self.min_v_degree,
            "max_v_degree": self.max_v_degree,
        }

        self.edges_info = {
            "#internal_edges": self.num_internal_edges,
            "#boundary_edges": self.num_boundary_edges,
            "#nonmanifold_edges": self.num_nonmanifold_edges,
            "#nonmanifold_vertices": self.num_nonmanifold_vertices,
            "min_connectivity": self.min_connectivity,
            "max_connectivity": self.max_connectivity,
            "avg_connectivity": self.avg_connectivity,
            "min_edge_length[mel]": self.min_edge_length,
            "max_edge_length[mal]": self.max_edge_length,
            "aspect_ratio[ar][mal/mel]": self.edge_aspect_ratio,
        }

        self.faces_info = {
            "#intersected_faces": self.num_intersected_faces,
            "#intersected_pairs": self.num_intersected_pairs,
            "intersected_faces_ratio[%]": self.intersected_faces_ratio_pct,
            "#degenerate_faces": self.num_degenerate_faces,
            "#non_degenerate_faces": self.num_nondegenerate_faces,
            "min_f_angle[rad]": self.min_f_angle_rad,
            "max_f_angle[rad]": self.max_f_angle_rad,
            "min_f_angle[deg]": self.min_f_angle_deg,
            "max_f_angle[deg]": self.max_f_angle_deg,
            "min_f_area": self.min_f_area,
            "max_f_area": self.max_f_area,
            "min_dihedral_angle[deg]": self.min_dihedral_angle_deg,
            "max_dihedral_angle[deg]": self.max_dihedral_angle_deg,
            "num_dup_faces": self.num_dup_faces,
            "f_angle_distribution[deg]": self.face_angles_deciles,
            "f_area_distribution": self.face_areas_deciles,
            "dihedral_angle_distribution[deg]": self.face_adjacency_angles_deciles,
        }

        self.ccs_info = {
            "#ccs": self.body_count,
            "#ccs[split][wt=True]": self.num_ccs_wt,
            "#ccs[split][wt=False]": self.num_ccs_nwt,
            "ccs_max_f_wt" : self.ccs_max_f_wt,
            "ccs_max_f_non_wt" : self.ccs_max_f_nwt,
            "ccs_min_f_wt" : self.ccs_min_f_wt,
            "ccs_min_f_non_wt" : self.ccs_min_f_nwt,
            "css_max_v_wt" : self.ccs_max_v_wt,
            "css_max_v_non_wt" : self.ccs_max_v_nwt,
            "css_min_v_wt" : self.ccs_min_v_wt,
            "css_min_v_non_wt" : self.ccs_min_v_nwt,
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
        info_str += f"  {Fore.CYAN}{'#intersected_faces / #pairs / ratio[%]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['#intersected_faces'])} / "
        info_str += f"{format_value(self.faces_info['#intersected_pairs'])} / "
        info_str += f"{format_value(self.faces_info['intersected_faces_ratio[%]'])}\n"
        
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

        info_str += f"  {Fore.CYAN}{'f_angle dist[10%-90%][deg]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += " / ".join([f"{v:.1f}" for v in self.face_angles_deciles.values()]) + "\n"
        
        info_str += f"  {Fore.CYAN}{'f_area dist[10%-90%]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += " / ".join([f"{v:.4e}" for v in self.face_areas_deciles.values()]) + "\n"

        info_str += f"  {Fore.CYAN}{'dihedral_angle dist[10%-90%][deg]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += " / ".join([f"{v:.1f}" for v in self.face_adjacency_angles_deciles.values()]) + "\n"

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

    def to_dict(self, nested=False) -> dict:

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

        return info_dict
