# Mesh Analysis and Visualization for Research

A powerful 3D mesh and point cloud analysis and visualization toolkit built with OpenGL and Python. Load, inspect, and analyze 3D mesh and point cloud files with real-time rendering and comprehensive geometric analysis.

## Overview

This project provides two complementary visualization applications for 3D data research and analysis:

| Application | Purpose | Formats | Use Cases |
|---|---|---|---|
| **`main_mesh.py`** (Mesh Viewer) | Full topology analysis and mesh visualization | OBJ, STL, PLY, GLB, OFF | Mesh validation, defect detection, multi-mesh comparison |
| **`main_pc.py`** (Point Cloud Viewer) | Large-scale point cloud visualization | XYZ | LiDAR analysis, point cloud inspection, multi-cloud comparison |

## Installation

### Requirements
- Python 3.7+
- OpenGL 3.3+ capable graphics card
- 2GB+ RAM (4GB+ for large point clouds)

### Setup

**Platform-Specific Environment Setup:**

```bash
# For macOS
conda env create -f env_yaml/mac_env.yml
conda activate meshviewer

# For Windows
conda env create -f env_yaml/win_env.yml
conda activate meshviewer
```

**Or install dependencies manually:**
```bash
pip install glfw PyOpenGL numpy pyrr pillow trimesh python-fcl colorama
```

**Install Python Package:**
```bash
pip install git+https://github.com/KhoiDOO/meshinfo.git
```

## Getting Started

### Mesh Viewer
```bash
python main_mesh.py
# Press O to open a mesh file
# See docs/MESH_VIEWER.md for full documentation
```

### Point Cloud Viewer
```bash
python main_pc.py
# Press O to open a point cloud file
# See docs/POINT_CLOUD_VIEWER.md for full documentation
```

## Features at a Glance

### Mesh Viewer (main_mesh.py)
- ✅ Multi-mesh loading with automatic grid layout
- ✅ Topology analysis: self-intersections, non-manifold detection
- ✅ Visualization: face/vertex normals, point clouds, wireframe overlays
- ✅ Comprehensive mesh statistics in console output
- ✅ Side-by-side mesh comparison

### Point Cloud Viewer (main_pc.py)
- ✅ Multi-cloud loading with synchronized views
- ✅ GPU-optimized rendering for millions of points
- ✅ Support for colored point clouds (RGB)
- ✅ Adaptive point sizing and camera controls
- ✅ Multi-format support (XYZ, LAS, LAZ, PLY)

## Documentation

Detailed documentation for each application:

- **[Mesh Viewer Documentation](docs/MESH_VIEWER.md)** - Full guide for `main_mesh.py`
  - Mesh topology analysis features
  - Keyboard controls and usage workflow
  - Supported formats and sample meshes
  - Performance optimization tips

- **[Point Cloud Viewer Documentation](docs/POINT_CLOUD_VIEWER.md)** - Full guide for `main_pc.py`
  - Point cloud loading and visualization
  - Keyboard controls and camera management
  - Format specifications and examples
  - Creating point cloud files from various sources

## Sample Data

Both applications include sample data for testing:

- **Mesh Samples** (`samples/mesh/`)
  - Test meshes with known topology issues
  - Good for validation and demonstration
  - See [samples/mesh/README.md](samples/mesh/README.md)

- **Point Cloud Samples** (`samples/pc/`)
  - Example point clouds in XYZ format
  - Ready to load and visualize
  - See [samples/pc/README.md](samples/pc/README.md)

## Architecture

### Technologies Used
- **Rendering**: OpenGL 3.3 Core Profile with GLSL shaders
- **Mesh Processing**: Trimesh library for geometry operations
- **Collision Detection**: FCL (Flexible Collision Library) BVH
- **GUI**: GLFW for window management and input
- **File I/O**: NumPy, Trimesh, Laspy (optional)

## License

See [LICENSE](LICENSE) file for details.