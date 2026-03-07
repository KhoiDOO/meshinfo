import trimesh
import numpy as np
import fcl
import warnings
from colorama import Fore, Style, init
from . import format_value, format_bool

from ..constants import (
    COPLANAR_TOLERANCE,
    MANIFOLD_EDGE_COUNT,
    FORMAT_LABEL_WIDTH,
    FORMAT_PRECISION_FLOAT,
    FORMAT_PRECISION_COORD
)

from trimesh.triangles import mass_properties, MassProperties

# Initialize colorama for Windows compatibility
init(autoreset=True)

def is_manifold(mesh: trimesh.Trimesh) -> bool:
    edges_sorted = np.sort(mesh.edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    is_manifold = np.all(counts == MANIFOLD_EDGE_COUNT).item()
    return is_manifold

def get_intersected_tria_ids(mesh: trimesh.Trimesh):
    # 1. Build the FCL Model
    model = fcl.BVHModel()
    model.beginModel(len(mesh.vertices), len(mesh.faces))
    model.addSubModel(mesh.vertices, mesh.faces)
    model.endModel()

    mesh_obj = fcl.CollisionObject(model, fcl.Transform())

    # 2. Collision Request
    request = fcl.CollisionRequest(enable_contact=True, num_max_contacts=len(mesh.faces) ** 2)
    result = fcl.CollisionResult()
    fcl.collide(mesh_obj, mesh_obj, request, result)

    intersected_ids = set()

    # 3. The "Zero Shared Vertices" Filter
    for contact in result.contacts:
        id1, id2 = contact.b1, contact.b2

        if id1 == id2:
            continue

        # Get the vertex indices for both triangles
        v1 = set(mesh.faces[id1])
        v2 = set(mesh.faces[id2])

        # INTERSECTION LOGIC:
        # If they share 1 or more vertices, they are "touching" (neighbors).
        # We only care if they share 0 vertices AND FCL says they collide.
        if len(v1.intersection(v2)) == 0:
            intersected_ids.add(id1)
            intersected_ids.add(id2)

    return list(intersected_ids)


def get_nonmanifold_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    nonmanifold_vertices = set()
    
    # First, collect vertices on non-manifold edges
    edges_unique, edges_counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    nonmanifold_edge_mask = edges_counts > 2  # Edges shared by more than 2 faces
    if np.any(nonmanifold_edge_mask):
        nonmanifold_edge_vertices = edges_unique[nonmanifold_edge_mask].flatten()
        nonmanifold_vertices.update(nonmanifold_edge_vertices)
    
    # Preprocess face adjacency into a more efficient lookup structure
    # Build a dictionary: face_id -> list of adjacent face_ids
    face_adjacency_dict: dict[int, list[int]] = {}
    for face_pair in mesh.face_adjacency:
        f1, f2 = face_pair[0], face_pair[1]
        if f1 not in face_adjacency_dict:
            face_adjacency_dict[f1] = []
        if f2 not in face_adjacency_dict:
            face_adjacency_dict[f2] = []
        face_adjacency_dict[f1].append(f2)
        face_adjacency_dict[f2].append(f1)
    
    # For each vertex, check if its adjacent faces form a single connected component
    for vertex_idx in range(len(mesh.vertices)):
        # Get all faces adjacent to this vertex
        adjacent_faces = np.where(np.any(mesh.faces == vertex_idx, axis=1))[0]
        
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
    def __init__(self, mesh: trimesh.Trimesh, name: str = "Mesh"):
        self.mesh = mesh
        self.name = name
        self.intersected_face_ids = get_intersected_tria_ids(mesh)
        self.non_watertight_components = mesh.split(only_watertight=False)
        self.watertight_components = mesh.split(only_watertight=True)
        self.genus = 1 - (len(mesh.vertices) - len(mesh.edges_unique) + len(mesh.faces)) / 2
        self.num_dup_faces = get_num_dup_faces(mesh)

        self.vertex_defects: np.ndarray = mesh.vertex_defects
        self.vertex_degree: np.ndarray = mesh.vertex_degree

        self.edges_unique: np.ndarray
        self.edges_counts: np.ndarray
        self.edges_unique, self.edges_counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        self.vertex_connectivity = np.bincount(self.edges_unique.flatten(), minlength=len(mesh.vertices))
        self.edges_unique_length: np.ndarray
        self.edges_unique_length = self.mesh.edges_unique_length
        
        self.nonmanifold_edge_mask = self.edges_counts > 2  # Edges shared by more than 2 faces
        self.nonmanifold_edges = self.edges_unique[self.nonmanifold_edge_mask]
        self.num_nonmanifold_edges = np.sum(self.nonmanifold_edge_mask).item()
        
        self.nonmanifold_vertices = get_nonmanifold_vertices(mesh)
        self.num_nonmanifold_vertices = len(self.nonmanifold_vertices)

        self.nondegenerate_faces_mask = mesh.nondegenerate_faces()
        self.num_degenerate_faces = np.sum(~self.nondegenerate_faces_mask).item()
        self.num_nondegenerate_faces = np.sum(self.nondegenerate_faces_mask).item()
        self.face_angles: np.ndarray = mesh.face_angles
        self.face_areas: np.ndarray = mesh.area_faces
        self.face_adjacency_angles: np.ndarray = mesh.face_adjacency_angles
        
        self.stats = {
            "#vertices": len(mesh.vertices),
            "#faces": len(mesh.faces),
            "#edges": len(mesh.edges_unique),
            "genus": self.genus,
            "#components": mesh.body_count,
            "#components[split][watertight=True]": len(self.watertight_components),
            "#components[split][non_watertight=True]": len(self.non_watertight_components),
        }

        self.properties = {
            "is_watertight": mesh.is_watertight,
            "is_empty": mesh.is_empty,
            "is_winding_consistent": mesh.is_winding_consistent,
            "is_convex": mesh.is_convex,
            "is_manifold[ignore intersection]": is_manifold(mesh),
            "is_manifold": is_manifold(mesh) and len(self.intersected_face_ids) == 0,
            "mutable": mesh.mutable,
            "is_intersecting": len(self.intersected_face_ids) > 0,
        }

        volume, center_mass = get_volume_center_mass_density(mesh.triangles)
        self.analysis = {
            "area": mesh.area,
            "volume": volume,
            "sphericity": get_sphericity(volume, mesh.area),
            "bounds": mesh.bounds,
            "center_mass": center_mass,
            "centroid": mesh.centroid,
            "extents": mesh.extents,
        }

        self.vertices_info = {
            "#coplanar_vertices": np.sum(np.abs(self.vertex_defects) < COPLANAR_TOLERANCE).item(),
            "#convex_vertices": np.sum(self.vertex_defects > 0).item(),
            "#concave_vertices": np.sum(self.vertex_defects < 0).item(),
            "min_vertex_degree": int(self.vertex_degree.min()),
            "max_vertex_degree": int(self.vertex_degree.max()),
        }

        self.edges_info = {
            "#internal_edges": np.sum(self.edges_counts == 2).item(),
            "#boundary_edges": np.sum(self.edges_counts == 1).item(),
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
            "#intersected_faces": len(self.intersected_face_ids),
            "#degenerate_faces": self.num_degenerate_faces,
            "#non_degenerate_faces": self.num_nondegenerate_faces,
            "min_face_angle[rad]": float(self.face_angles.min()),
            "max_face_angle[rad]": float(self.face_angles.max()),
            "min_face_angle[deg]": float(np.degrees(self.face_angles.min())),
            "max_face_angle[deg]": float(np.degrees(self.face_angles.max())),
            "min_face_area": float(self.face_areas.min()),
            "max_face_area": float(self.face_areas.max()),
            "min_dihedral_angle[deg]": float(np.degrees(np.min(self.face_adjacency_angles))) if len(self.face_adjacency_angles) > 0 else None,
            "max_dihedral_angle[deg]": float(np.degrees(np.max(self.face_adjacency_angles))) if len(self.face_adjacency_angles) > 0 else None,
            "num_dup_faces": self.num_dup_faces,
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
        # Row 1: Topological properties
        info_str += f"  {Fore.CYAN}watertight:{Style.RESET_ALL} {format_bool(self.properties['is_watertight'])}  "
        info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}manifold (ignore intersection):{Style.RESET_ALL} {format_bool(self.properties['is_manifold[ignore intersection]'])}  {Fore.CYAN}manifold:{Style.RESET_ALL} {format_bool(self.properties['is_manifold'])}  "
        info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}winding_consistent:{Style.RESET_ALL} {format_bool(self.properties['is_winding_consistent'])}\n"
        
        # Row 2: Geometric and state properties
        info_str += f"  {Fore.CYAN}convex:{Style.RESET_ALL} {format_bool(self.properties['is_convex'])}  "
        info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}empty:{Style.RESET_ALL} {format_bool(self.properties['is_empty'])}  "
        info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}intersecting:{Style.RESET_ALL} {format_bool(self.properties['is_intersecting'])}  "
        info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}mutable:{Style.RESET_ALL} {format_bool(self.properties['mutable'])}\n"
        
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Analysis:{Style.RESET_ALL}\n"
        for key, value in self.analysis.items():
            if key == "bounds":
                value_str = f"{Fore.YELLOW}[[{value[0][0]:.{FORMAT_PRECISION_COORD}f}, {value[0][1]:.{FORMAT_PRECISION_COORD}f}, {value[0][2]:.{FORMAT_PRECISION_COORD}f}], "
                value_str += f"[{value[1][0]:.{FORMAT_PRECISION_COORD}f}, {value[1][1]:.{FORMAT_PRECISION_COORD}f}, {value[1][2]:.{FORMAT_PRECISION_COORD}f}]]{Style.RESET_ALL}"
                info_str += f"  {Fore.CYAN}{key:.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {value_str}\n"
            elif key == "extents":
                value_str = f"{Fore.YELLOW}[l = {value[0]:.{FORMAT_PRECISION_COORD}f}, w = {value[1]:.{FORMAT_PRECISION_COORD}f}, h = {value[2]:.{FORMAT_PRECISION_COORD}f}]{Style.RESET_ALL}"
                info_str += f"  {Fore.CYAN}{key:.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {value_str}\n"
            else:
                formatted_value = format_value(value)
                info_str += f"  {Fore.CYAN}{key:.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {formatted_value}\n"
        
        # Vertices Info - group related items
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Vertices Info:{Style.RESET_ALL}\n"
        info_str += f"  {Fore.CYAN}{'#coplanar / #convex / #concave':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.vertices_info['#coplanar_vertices'])} / "
        info_str += f"{format_value(self.vertices_info['#convex_vertices'])} / "
        info_str += f"{format_value(self.vertices_info['#concave_vertices'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max vertex_degree':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.vertices_info['min_vertex_degree'])} / "
        info_str += f"{format_value(self.vertices_info['max_vertex_degree'])}\n"
        
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
        info_str += f"{format_value(self.faces_info['min_face_angle[rad]'])} / "
        info_str += f"{format_value(self.faces_info['max_face_angle[rad]'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max face_angle[deg]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['min_face_angle[deg]'])} / "
        info_str += f"{format_value(self.faces_info['max_face_angle[deg]'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max face_area':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['min_face_area'])} / "
        info_str += f"{format_value(self.faces_info['max_face_area'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max dihedral_angle[deg]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['min_dihedral_angle[deg]'])} / "
        info_str += f"{format_value(self.faces_info['max_dihedral_angle[deg]'])}\n"

        info_str += f"  {Fore.CYAN}{'num_dup_faces':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.faces_info['num_dup_faces'])}\n"
        
        info_str += f"\n{Fore.CYAN}{Style.BRIGHT}╚═══════════════════════╝{Style.RESET_ALL}"
        return info_str

    def to_dict(self, np2list=True):

        info_dict = {
            **self.stats,
            **self.properties,
            **self.analysis,
            **self.vertices_info,
            **self.edges_info,
            **self.faces_info,
        }

        if np2list:
            for key, value in info_dict.items():
                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        info_dict[key] = value.item()
                    info_dict[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    info_dict[key] = value.item()

        return info_dict