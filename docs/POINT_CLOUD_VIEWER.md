# Point Cloud Viewer (`main_pc.py`)

Optimized 3D point cloud visualization tool for large-scale point cloud data with real-time GPU rendering.

## Quick Start

```bash
python main_pc.py
```

Then press **O** to open a point cloud file.

## Features

### Point Cloud Loading & Formats
- Support for multiple point cloud formats: **XYZ**, **LAS**, **LAZ**, **PLY**
- **Multi-cloud loading**: Load and compare multiple point clouds simultaneously
- Automatic grid layout for multiple point clouds
- Automatic point cloud normalization
- Efficient memory management for large datasets

### Real-time Rendering
- **GPU Optimized**: Custom streaming pipeline for millions of points
- **Adaptive Point Size**: Automatic sizing based on cloud density
- **Color Mapping**: Native support for RGB color per point
- **Spatial Bounds**: Visual indication of point cloud extents

### Visualization Features
- **Automatic Centering**: All clouds normalized to standard volume
- **Multi-cloud Comparison**: Side-by-side analysis of multiple clouds
- **Point Size Control**: Adjust point rendering size for clarity
- **Camera Controls**: Orbit, pan, and zoom with smooth transitions
- **Color Themes**: Dark and light backgrounds for clarity

### Color Themes
- **Dark Theme**: Dark background with light points (default)
- **Light Theme**: Light background with dark points
- **Per-Point Colors**: RGB values preserved from XYZ format

### Screenshot Export
- Capture current view to image file
- Export formats: **PNG**, **JPEG**, **PDF**
- File dialog for save location selection

## Keyboard Controls

| Key | Action | Description |
|-----|--------|-------------|
| **O** | Open File | Open file dialog to load point cloud file(s) - supports multiple selection |
| **Tab** | Open & Append | Open file dialog to load point cloud file(s) and append it to the current point buffer |
| **P** | Cycle Point Size | Adjust point rendering size for better visibility |
| **U** | Toggle Color Theme | Switch between dark and light background themes |
| **SPACE** | Toggle Camera Rotation | Switch between automatic rotation and manual control |
| **R** | Reset Camera | Reset camera to default position |
| **W** | Move Forward | Move camera forward in view direction |
| **S** | Move Backward | Move camera backward in view direction |
| **A** | Strafe Left | Move camera left |
| **D** | Strafe Right | Move camera right |
| **Q** | Roll Left | Rotate view around forward axis |
| **E** | Roll Right | Rotate view around forward axis |
| **Z** | Scale Down | Reduce point size scale factor |
| **X** | Scale Up | Increase point size scale factor |
| **↑** | Move Up | Raise camera position |
| **↓** | Move Down | Lower camera position |

## Usage Workflow

1. **Load point clouds**: Press **O** to open file dialog
   - Select single cloud for detailed viewing
   - Select multiple clouds (Cmd/Ctrl+click) for comparison

2. **Set color theme**: Press **U** to toggle theme
   - Use dark theme for detailed point viewing
   - Use light theme for presentations

4. **Analyze spatial distribution**: Use camera controls
   - **W/S/A/D** - Move camera through space
   - Rotate view to inspect cloud structure
   - Identify density variations and patterns

5. **Compare multiple clouds**: Use grid layout
   - Each cloud automatically positioned in grid
   - All cameras synchronized
   - Visually compare structure and coverage

## Supported Point Cloud Formats

### XYZ Format (Plain Text)
**File extension**: `.xyz`
**Format**: One point per line, space or tab-separated

```
X Y Z
10.5 20.3 15.2
11.2 21.1 15.8
```

With RGB color:
```
X Y Z R G B
10.5 20.3 15.2 255 0 0
11.2 21.1 15.8 0 255 0
```

**Advantages**:
- Human-readable
- Easy to generate and debug
- Universal compatibility
- Supports color information

**Disadvantages**:
- Large file size for millions of points
- No compression support
- No metadata storage

### LAS Format (Binary)
**File extensions**: `.las`, `.laz`
**Description**: Industry standard for LiDAR point cloud data

**Characteristics**:
- Binary format with optional LAZ compression
- Rich metadata (intensity, classification, returns)
- Efficient storage (6-20 bytes per point)
- Wide tool support in geospatial community

**Advantages**:
- Compact file size
- Industry standard
- Compression support
- Preserves LiDAR attributes

**Requirements**:
- Install `laspy`: `pip install laspy`

### PLY Format
**File extension**: `.ply`
**Description**: Polygon file format (also used for point clouds)

**Characteristics**:
- Flexible ASCII or binary encoding
- Support for arbitrary attributes per point
- Standard header with format specification

**Advantages**:
- Flexible attribute support
- Both text and binary options
- Good for mixed mesh + cloud workflows

## Sample Point Clouds

The `samples/pc/` folder contains test point clouds:

- **`airplane.xyz`** - Sample point cloud in XYZ format
  - Simple ASCII format
  - Good for basic testing
  - No color information (default white)

See [samples/pc/README.md](../samples/pc/README.md) for detailed information.

## Loading Point Clouds from Files

```bash
# Load single cloud
python main_pc.py
# Then press O and select a file

# Supported in file dialog
# - .xyz files (recommended for testing)
# - .las / .laz files (production LiDAR data)
# - .ply files (can be point clouds or meshes)
```

## Creating Point Cloud Files

### From XYZ Text
```bash
# Create simple points.xyz
cat > points.xyz << EOF
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
EOF
```

### From NumPy (Python)
```python
import numpy as np

# Generate random point cloud
points = np.random.rand(10000, 3) * 100

# Save to XYZ format
np.savetxt('pointcloud.xyz', points, fmt='%.6f')

# With RGB color
colors = np.random.randint(0, 256, (10000, 3))
points_with_color = np.hstack([points, colors])
np.savetxt('colored.xyz', points_with_color, 
           fmt='%.6f %.6f %.6f %d %d %d')
```

### From LiDAR Scans
```python
import open3d as o3d

# Load LAS file
pcd = o3d.io.read_point_cloud("lidar_scan.las")

# Export to XYZ
o3d.io.write_point_cloud("lidar_scan.xyz", pcd)

# Filter and downsample
pcd_down = pcd.voxel_down_sample(voxel_size=0.01)
o3d.io.write_point_cloud("downsampled.xyz", pcd_down)
```

## Technical Architecture

### Core Components
- **Point Cloud Buffer Management**: GPU memory streaming for large datasets
- **Adaptive Rendering Pipeline**: LOD-based point sizing
- **Format Loaders**: XYZ, LAS, PLY parsers
- **Spatial Analysis**: Bounding box computation, density estimation

### GPU Rendering
- OpenGL 3.3 Core Profile with GLSL shaders
- GL_POINTS primitives for efficient rendering
- Programmable point size based on density
- Direct GPU streaming for >1M points

### Performance Characteristics
- **Million points**: ~60 FPS (smooth rotation)
- **10M points**: ~30 FPS (interactive)
- **100M points**: ~6 FPS (static viewing)
- Depends on GPU and display resolution

## Performance Tips

**For Very Large Clouds (>50M points):**
- Downsample before loading: `voxel_down_sample()` in open3d
- Use LAS format with LAZ compression
- Disable unnecessary visualizations
- Run on GPU with 4GB+ VRAM

**For Multiple Clouds:**
- Downsample each cloud to <1M points
- Use consistent coordinate systems
- Load similar-density clouds for balanced layout

**For Real-time Interaction:**
- Use point size **P** to adjust visibility
- Disable auto-rotation with **SPACE**
- Smooth camera controls with **W/A/S/D**

## Troubleshooting

**File won't load:**
- Check file format is supported (XYZ, LAS/LAZ, PLY)
- Verify file path has no special characters
- Check file encoding is UTF-8 for XYZ files
- Try manual coordinate normalization if bounds too large

**Points appear as artifacts:**
- Check coordinate ranges (may need clipping)
- Verify point format matches selected file type
- Try with smaller point cloud first
- Check for NaN or infinite values in data

**Poor performance with large clouds:**
- Reduce point count using `voxel_down_sample()`
- Use LAS format with compression
- Increase GPU memory allocation
- Run with higher-end GPU (RTX series recommended)

**Graphics artifacts or flickering:**
- Update GPU drivers
- Verify OpenGL 3.3+ support
- Try adjusting point size with **P** key
- Check system GPU resources (Task Manager/Activity Monitor)

## Memory Requirements

| Cloud Size | XYZ File | LAS File | GPU Memory |
|-----------|----------|----------|-----------|
| 1M points | 24 MB | 4-8 MB | 16 MB |
| 10M points | 240 MB | 40-80 MB | 160 MB |
| 100M points | 2.4 GB | 400-800 MB | 1.6 GB |

Estimates for single cloud. Multiple clouds require proportional memory.
