# Mesh Viewer (`main_mesh.py`)

Full-featured 3D mesh visualization and topology analysis tool built with OpenGL and Python.

## Quick Start

```bash
python main_mesh.py
```

Then press **O** to open a mesh file.

## Features

### Mesh Loading & Formats
- Support for multiple 3D mesh formats: **OBJ**, **STL**, **PLY**, **GLB**, **OFF**
- **Multi-mesh loading**: Load and compare multiple meshes simultaneously
- Automatic grid layout for multiple meshes with adjustable spacing
- Automatic mesh normalization and centering
- Scene concatenation for multi-object files

### Rendering Modes
- **Solid Mode**: Fill rendering with lighting
- **Wireframe Mode**: Edge-based visualization  
- **Combined Mode**: Solid + wireframe overlay
- Automatic camera rotation around mesh
- Manual object rotation and scaling controls

### Mesh Analysis & Visualization
- **Intersected Faces Detection**: Automatically detects and highlights self-intersecting triangles
- **Face Normals Display**: Visualize per-face normal vectors (green)
- **Vertex Normals Display**: Visualize per-vertex normal vectors (blue)
- **Point Cloud View**: Sample-based point cloud representation (yellow)
- **Point Cloud Normals Display**: Visualize normals at sampled point cloud positions (magenta)
- **Non-Manifold Edges Display**: Highlight edges shared by more than two faces
- **Non-Manifold Vertices Display**: Highlight vertices with invalid local topology

### Mesh Information Display
Displays comprehensive mesh statistics in the console:
- **Statistics**: Vertex/face/edge count, genus, component count
- **Properties**: Watertight, manifold, convex, winding consistency, self-intersection status
- **Analysis**: Area, volume, bounds, center of mass, extents
- **Edge Info**: Internal/boundary/non-manifold edges, non-manifold vertices, connectivity stats, edge lengths, aspect ratio
- **Face Info**: Intersected faces, degenerate faces

### Color Themes
- **Dark Theme**: Dark background with light mesh colors (default)
- **Light Theme**: Light background with dark mesh colors
- Dynamic color adaptation for all visualization elements

### Screenshot Export
- Capture current view to image file
- Auto-crop to remove empty space
- Export formats: **PNG**, **JPEG**, **PDF**
- File dialog for save location selection

### Platform Dialogs
- Native macOS open/save dialogs via `osascript`
- Tk-based file dialogs as a cross-platform fallback

## Keyboard Controls

| Key | Action | Description |
|-----|--------|-------------|
| **O** | Open File | Open file dialog to load mesh file(s) - supports multiple selection |
| **Tab** | Open & Append | Open file dialog to load mesh file(s) and append it to the current mesh buffer |
| **J** | Solid Mode | Render mesh with filled polygons |
| **K** | Wireframe Mode | Render mesh with edges only |
| **L** | Combined Mode | Render both solid and wireframe |
| **I** | Toggle Intersected Faces | Highlight self-intersecting faces in orange |
| **N** | Toggle Face Normals | Show/hide per-face normal vectors (green) |
| **M** | Toggle Vertex Normals | Show/hide per-vertex normal vectors (blue) |
| **P** | Toggle Point Cloud | Show/hide sampled point cloud (yellow, 8192 points) |
| **Y** | Toggle Point Cloud Normals | Show/hide point cloud normal vectors (magenta) |
| **H** | Toggle Non-Manifold Edges | Show/hide non-manifold edge highlights |
| **V** | Toggle Non-Manifold Vertices | Show/hide non-manifold vertex highlights |
| **U** | Toggle Color Theme | Switch between dark and light theme |
| **C** | Capture Screenshot | Save current view as PNG/JPEG/PDF |
| **SPACE** | Toggle Camera Rotation | Switch between automatic rotation and manual control |
| **R** | Reset Camera | Reset camera to default position and angle |
| **W** | Rotate Object Up | Rotate object upward (X axis) |
| **A** | Rotate Object Left | Rotate object left (Y axis) |
| **S** | Rotate Object Down | Rotate object downward (X axis) |
| **D** | Rotate Object Right | Rotate object right (Y axis) |
| **Q** | Roll Object Left | Roll object left (Z axis) |
| **E** | Roll Object Right | Roll object right (Z axis) |
| **Z** | Scale Down | Scale object down |
| **X** | Scale Up | Scale object up |
| **[** | Decrease Layout Spacing | Reduce spacing between meshes in multi-mesh view |
| **]** | Increase Layout Spacing | Increase spacing between meshes in multi-mesh view |
| **↑** | Move Camera Up | Increase camera height |
| **↓** | Move Camera Down | Decrease camera height |

## Usage Workflow

1. **Load meshes**: Press **O** to open file dialog
   - Select single file for detailed analysis
   - Select multiple files (Cmd/Ctrl+click) for side-by-side comparison

2. **Switch rendering modes**: Use **J/K/L** to change visualization
   - **J** (Solid) - See filled geometry
   - **K** (Wireframe) - See edge structure
   - **L** (Both) - See both solid and edges

3. **Analyze topology**: Press visualization toggles
   - **I** - Highlight self-intersecting faces (orange)
   - **H** - Highlight non-manifold edges (red)
   - **V** - Highlight non-manifold vertices (cyan)

4. **Inspect normals**: Press **N** or **M**
   - **N** - Show face normals (green lines)
   - **M** - Show vertex normals (blue lines)
   - Helps identify normal direction issues

5. **Visualize point sampling**: Press **P** and **Y**
   - **P** - Show point cloud sampling (yellow)
   - **Y** - Show point cloud normals (magenta)

6. **Manipulate view**: Use mouse and keyboard
   - **W/A/S/D/Q/E** - Rotate object
   - **Z/X** - Scale object up/down
   - **SPACE** - Toggle automatic camera rotation
   - **R** - Reset to default view

7. **Export results**: Press **C** to save screenshot

## Sample Meshes

The `samples/mesh/` folder contains test meshes for validation:

- **`(non self-intersected) dumbbel.obj`** - Clean manifold mesh
  - Expected: `is_manifold: True`, `#intersected_faces: 0`
  - Use to verify proper handling of valid meshes

- **`(non-manifold edge) dolphin.obj`** - Non-manifold edges
  - Expected: `#nonmanifold_edges: > 0`
  - Press **H** to highlight problematic edges

- **`(self-intersected) hammer.obj`** - Self-intersecting faces
  - Expected: `is_intersecting: True`, `#intersected_faces: > 0`
  - Press **I** to highlight intersecting triangles

**Multi-Mesh Comparison**: Load all three samples together to compare topology side-by-side.

See [samples/mesh/README.md](../samples/mesh/README.md) for detailed information.

## Supported Mesh Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| Wavefront OBJ | `.obj` | Widely supported, text-based, no topology info |
| STereoLithography | `.stl` | Simple triangulated surface, binary or ASCII |
| PLY | `.ply` | Polygon file format, flexible, supports colors |
| glTF Binary | `.glb` | Modern format with materials, optimized |
| Object File Format | `.off` | Simple format with vertices and faces |

## Technical Architecture

### Core Components
- **`mesh.py`** - Mesh topology analysis and MeshInfo class
- **`mesh_buffer.py`** - GPU buffer management for mesh rendering
- **`constants.py`** - Global configuration and visualization constants

### Key Algorithms
- **Self-Intersection Detection**: FCL (Flexible Collision Library) BVH collision detection
- **Non-Manifold Edge Detection**: Counting edge usage across faces
- **Non-Manifold Vertex Detection**: Checking if adjacent faces form connected components
- **Mesh Analysis**: Genus calculation, defect computation, connectivity analysis

### GPU Rendering
- OpenGL 3.3 Core Profile with GLSL shaders
- Separate buffers for mesh, intersections, normals, point clouds
- Polygon offset for clean wireframe overlays
- Adaptive normal length based on mesh size

## Troubleshooting

**Mesh won't load:**
- Check file format is supported (OBJ, STL, PLY, GLB, OFF)
- Verify file path has no special characters
- Try opening from command line for error messages

**Graphics artifacts:**
- Update GPU drivers
- Verify OpenGL 3.3+ support
- Try disabling advanced visualizations

**Slow performance:**
- Reduce mesh complexity using external tools
- Disable all toggles except needed ones
- Try solid mode instead of wireframe

**Console shows warnings:**
- Check mesh has valid face connectivity
- Verify vertices are in reasonable coordinate ranges
- Look for degenerate triangles in analysis output
