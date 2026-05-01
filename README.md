# Mesh Analysis and Visualization for Research

A powerful 3D mesh and point cloud analysis and visualization toolkit built with ModernGL and Python. Load, inspect, and analyze 3D mesh and point cloud files with real-time rendering and comprehensive geometric analysis.

## Overview

### Diagnostic & Comparison
| ![Analysis Dashboard](docs/dashboard.png) | ![Multi-mesh Comparison](docs/multiple_mesh.png) | ![Non-manifold Detection](docs/non_manifolde_vert_edge.png) |
|:---:|:---:|:---:|
| **Analysis Dashboard** | **Multi-mesh Comparison** | **Non-manifold Detection** |

### Visualization Modes
| ![Solid & Wireframe](docs/solid_wireframe.png) | ![Self-intersection](docs/intersection.png) | ![Point Cloud](docs/pointcloud.png) |
|:---:|:---:|:---:|
| **Solid & Wireframe Modes** | **Self-intersection** | **Point Cloud** |
| ![Face Normals](docs/faces_normal.png) | ![Vertex Normals](docs/vertices_normal.png) | ![Point Cloud Normals](docs/pointcloud_normal.png) |
| **Face Normals** | **Vertex Normals** | **Point Cloud Normals** |

## Installation

### Requirements
- Python 3.7+
- OpenGL 3.3+ capable graphics card

### Setup

**Running with Repo**
```bash
git clone https://github.com/KhoiDOO/meshinfo.git
cd meshinfo
```

**Create conda env**
```bash
conda create -n meshviewer python=3.10
conda activate meshviewer
pip install .
```

**Install as Python Package:**
```bash
pip install git+https://github.com/KhoiDOO/meshinfo.git
```

## Getting Started

### Mesh Viewer
```bash
python main.py
# Press O to open a mesh file
# See docs/VIEWER.md for full documentation
```
Enable mesh intersection checking (can be expensive on large meshes):
```bash
python main.py --intersect
```
Enable non-manifold vertex checking:
```bash
python main.py --nonmanifold
```
Enable additional analysis flags:
```bash
python main.py --components --geometry --topology
```

### Mesh Analysis API
In case you want to just analyze your mesh w/o viewing it, we provide a clean API.

```python
import trimesh
import os
from meshinfo import MeshInfo

mesh_path = "your_mesh_file.obj"
filename = os.path.basename(mesh_path).split(".")[0]
mesh = trimesh.load(mesh_path, process=False)

mesh_info = MeshInfo(
    mesh,
    name=filename,
    check_intersection=True,
    check_components=True,
    check_nonmanifold_vertices=True,
    check_geometry=True,
    check_topology=True,
    verbose=True
)

# Export all metrics as a dictionary
mesh_dict = mesh_info.to_dict(nested=True)
```

## Features at a Glance

### Mesh Viewer (`main.py`)
- ✅ **Interactive Analysis Dashboard**: Real-time side panel (Press **G** to toggle).
- ✅ **Comparison Table**: Side-by-side metrics for multiple meshes.
- ✅ **Advanced Camera Calibration**: Adjust FOV, Focal Length (35mm equivalent), and Clipping Planes.
- ✅ **Orbital Navigation**: Intuitive 3D rotation targeting the origin (Arrow keys).
- ✅ **Dynamic Layouts**: Automatically arrange multiple meshes in **Grid** or **Line** modes.
- ✅ **Topology Analysis**: Detect self-intersections, non-manifold edges/vertices, and genus.
- ✅ **Visualization**: Face/vertex normals, point clouds, and high-DPI screenshot export.

### Point Cloud Viewer (`main_pc.py`)
- ✅ Multi-cloud loading with synchronized views.
- ✅ GPU-optimized rendering for millions of points.
- ✅ Support for colored point clouds (RGB).
- ✅ Adaptive point sizing and camera controls.
- ✅ Multi-format support (XYZ, LAS, LAZ, PLY).

## Documentation

Detailed documentation for each application:

- **[Mesh Viewer Documentation](docs/VIEWER.md)** - Full guide for `main.py`
  - Camera calibration and navigation
  - Keyboard controls and usage workflow
  - Topology analysis algorithms
  - Performance optimization tips

## Sample Data

Both applications include sample data for testing:

- **Mesh Samples** (`samples/mesh`)
  - Test meshes with known topology issues (self-intersections, non-manifold edges).
  - See [samples/mesh/README.md](samples/mesh/README.md).

## Acknowledgement

This project is built upon several powerful open-source libraries:

- **Rendering**: [ModernGL](https://github.com/moderngl/moderngl) (OpenGL 3.3 Core Profile)
- **Geometry & Mesh Processing**: [Trimesh](https://github.com/mikedh/trimesh)
- **Self-Intersection Detection**: [MeshLib](https://github.com/MeshInspector/MeshLib)
- **GUI Framework**: [Dear ImGui](https://github.com/ocornut/imgui) (via [pyimgui](https://github.com/pyimgui/pyimgui))
- **Window Management**: [GLFW](https://github.com/glfw/glfw) (via [pyglfw](https://github.com/FlorianRhiem/pyglfw))
- **Numerical Computation**: [NumPy](https://github.com/numpy/numpy)
- **Math Utilities**: [Pyrr](https://github.com/approxion/pyrr)
- **Graph Theory**: [NetworkX](https://github.com/networkx/networkx) (for manifold analysis)
- **Image Processing**: [Pillow](https://github.com/python-pillow/Pillow) (for screenshots)
- **Terminal Styling**: [Colorama](https://github.com/tartley/colorama)

## License

See [LICENSE](LICENSE) file for details.
