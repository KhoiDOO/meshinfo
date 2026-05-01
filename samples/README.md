# Sample Meshes for Testing

This folder contains test mesh files with specific topology characteristics for demonstrating and validating the mesh analysis features of the viewer.

## Test Files

### 1. `(non self-intersected) dumbbel.obj`
**Characteristics:**
- Clean, manifold mesh
- No self-intersections
- No non-manifold edges or vertices
- Valid watertight topology

**Expected Analysis Results:**
- ✅ `is_watertight: True`
- ✅ `is_manifold: True`
- ✅ `is_intersecting: False`
- ✅ `#intersected_faces: 0`
- ✅ `#nonmanifold_edges: 0`
- ✅ `#nonmanifold_vertices: 0`

**Visualization Tests:**
- Press `I`: Should show no orange highlights (no intersections)
- Press `H`: Should show no edge highlights (no non-manifold edges)
- Press `V`: Should show no vertex highlights (no non-manifold vertices)

---

### 2. `(non-manifold edge) dolphin.obj`
**Characteristics:**
- Contains non-manifold edges (edges shared by more than 2 faces)
- May contain non-manifold vertices
- Invalid topology for watertight surfaces

**Expected Analysis Results:**
- ❌ `is_manifold: False` (or qualified as false)
- ⚠️ `#nonmanifold_edges: > 0`
- ⚠️ `#nonmanifold_vertices: > 0`

**Visualization Tests:**
- Press `H`: Highlights edges shared by more than 2 faces
- Press `V`: Highlights vertices with invalid local topology
- Use this to verify non-manifold detection algorithms

**Use Cases:**
- Testing edge-based topology validation
- Demonstrating vertex connectivity analysis
- Verifying non-manifold detection algorithms

---

### 3. `(self-intersected) hammer.obj`
**Characteristics:**
- Contains self-intersecting triangles
- Faces penetrate each other without sharing vertices
- Detected via FCL collision detection

**Expected Analysis Results:**
- ❌ `is_intersecting: True`
- ⚠️ `#intersected_faces: > 0`
- May still be manifold (edges properly connected) but geometrically invalid

**Visualization Tests:**
- Press `I`: Highlights intersecting triangles in orange
- Demonstrates FCL BVH collision detection
- Wireframe + fill visualizes the penetration

**Use Cases:**
- Testing self-intersection detection
- Validating FCL integration
- Demonstrating geometric vs. topological validity

---

## Multi-Mesh Comparison

Load all three meshes simultaneously to compare their topology side-by-side:

1. Press `O` to open file dialog
2. Hold `Cmd` (macOS) or `Ctrl` (Windows/Linux) and select all three files
3. Meshes will be arranged in an automatic grid layout
4. Use `[` / `]` to adjust spacing between meshes
5. Toggle visualization features to compare:
   - `I` - Compare intersection patterns
   - `H` - Compare non-manifold edges
   - `V` - Compare non-manifold vertices
   - `N` / `M` - Compare normal vectors
   - `P` - Compare point cloud sampling

## Technical Notes

- All meshes are automatically normalized to fit within a standard bounding volume
- Mesh names are extracted from filenames and displayed in console output
- Each mesh's analysis is printed independently to the console
- Visualization toggles apply to all loaded meshes simultaneously

## Adding Your Own Test Meshes

To add custom test meshes:
1. Place `.obj`, `.stl`, `.ply`, `.glb`, or `.off` files in this folder
2. Name files descriptively to indicate their characteristics (optional)
3. Load them via the `O` key in the viewer
4. Check console output for detailed topology analysis
