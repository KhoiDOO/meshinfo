# Mesh Viewer (`main.py`)

Full-featured 3D mesh visualization and topology analysis tool built with ModernGL and Python.

## Quick Start

```bash
python main.py
```

Then press **O** to open a mesh file.

## Features

### Mesh Loading & Formats
- Support for multiple 3D mesh formats: **OBJ**, **STL**, **PLY**, **GLB**, **OFF**
- **Multi-mesh loading**: Load and compare multiple meshes simultaneously
- **Dynamic Layouts**: Switch between **Grid** and **Line** arrangements for side-by-side comparison
- Automatic mesh normalization and centering
- Scene concatenation for multi-object files

### Analysis Dashboard (ImGui)
- **Real-time Side Panel**: Persistent dashboard for mesh statistics (Press **G** to toggle)
- **Comparison Table**: Side-by-side comparison of all loaded meshes
- **Detailed Metrics**: Expandable sections for Statistics, Topology, and Geometry
- **Distribution Plots**: Visual histograms for face angles and areas
- **Camera Calibration**: Dedicated tab for adjusting FOV, Focal Length, and Clipping Planes
- **Interactive Controls**: Toggle visualizations and adjust camera directly from the UI

### Rendering Modes
- **Solid Mode**: Fill rendering with lighting
- **Wireframe Mode**: Edge-based visualization  
- **Combined Mode**: Solid + wireframe overlay
- **Orbital Navigation**: Intuitive camera rotation that always targets the origin
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
Displays comprehensive mesh statistics in the console and UI:
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
- Capture current view to image file with auto-cropping
- Robust support for high-DPI (Retina) displays
- Export formats: **PNG**, **JPEG**, **PDF**

## Keyboard Controls

| Key | Action | Description |
|-----|--------|-------------|
| **O** | Open File | Open file dialog to load mesh file(s) - supports multiple selection |
| **G** | Toggle Dashboard | Show/hide the interactive analysis side panel |
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
| **SPACE** | Toggle Auto-Rotation | Switch between automatic horizontal rotation and manual control |
| **R** | Reset View | Reset camera and object to default positions |
| **Arrows (← →)** | Orbit Horizontally | Manually rotate the camera around the mesh (disables auto-rotation) |
| **Arrows (↑ ↓)** | Orbit Vertically | Manually rotate the camera up or down (disables auto-rotation) |
| **W/A/S/D/Q/E** | Rotate Object | Local rotation of the 3D model (independent of camera) |
| **Z / X** | Scale Model | Increase or decrease the global scale of the models |
| **[ / ]** | Layout Spacing | Adjust the distance between meshes in multi-mesh view |

## Usage Workflow

1. **Load meshes**: Press **O** to open file dialog. Select multiple files for side-by-side comparison.
2. **Switch Layout**: In the **Controls** tab of the dashboard, toggle between **Grid** and **Line** modes.
3. **Calibrate Camera**: Go to the **Camera** tab to set your preferred FOV or Focal Length.
4. **Analyze topology**: Use the visualization toggles (**I, H, V**) to highlight defects.
5. **Inspect normals**: Press **N** or **M** to identify normal direction issues.
6. **Navigate**: Use the **Arrow Keys** to orbit around your model and **Z/X** to zoom.
7. **Export**: Press **C** to save a screenshot of your analysis results.

## Technical Architecture

The project is structured into two main packages to separate analysis logic from visualization:

### `meshinfo` (Analysis Core)
- **`mesh.py`** - Deep topological and geometric analysis (watertight, manifold, self-intersections).
- **`constants.py`** - Analysis-specific tolerances and formatting.

### `viewer` (Visualization & UI)
- **`meshviewer.py`** - Main application logic and rendering loop.
- **`buffer.py`** - Efficient ModernGL buffer management for complex visualizations.
- **`constants.py`** - Shaders, color schemes, and UI settings.
- **`utils/io.py`** - High-level mesh loading and normalization.
- **`utils/fdialog.py`** - Cross-platform native file dialogs.

## Troubleshooting

**Mesh won't load:**
- Check file format is supported (OBJ, STL, PLY, GLB, OFF).
- Verify file path has no special characters.

**Graphics artifacts:**
- Update GPU drivers; ensure OpenGL 3.3+ support.

**Screenshot Issues:**
- On Retina/High-DPI displays, the tool automatically adjusts to the physical framebuffer size. Ensure you have the latest `Pillow` installed.
