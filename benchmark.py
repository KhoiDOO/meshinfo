import time
import argparse
import multiprocessing
from meshinfo.mesh import MeshInfo
from viewer.utils.io import load_mesh

def benchmark(mesh_path):
    print(f"--- Benchmarking: {mesh_path} ---")
    print(f"CPU Cores available: {multiprocessing.cpu_count()}")
    
    print("Loading mesh...")
    start_load = time.time()
    mesh = load_mesh(mesh_path)
    end_load = time.time()
    if mesh is None:
        print("Failed to load mesh.")
        return
    print(f"Mesh loaded in {end_load - start_load:.4f}s")
    print(f"Vertices: {len(mesh.vertices):,}")
    print(f"Faces: {len(mesh.faces):,}")

    # Run Analysis
    print("\nStarting MeshInfo analysis (all checks enabled)...")
    start_anal = time.time()
    info = MeshInfo(
        mesh,
        name="LargeMeshTest",
        check_components=True,
        check_intersection=True,
        check_nonmanifold_vertices=True,
        check_geometry=True,
        check_topology=True,
        verbose=True
    )
    end_anal = time.time()
    
    print("\n" + "="*50)
    print(f"Analysis Results for {info.name}:")
    print(f"  Total Analysis Time: {end_anal - start_anal:.4f}s")
    print(f"  Non-manifold Vertices: {info.num_nonmanifold_vertices}")
    print(f"  Body Count: {info.body_count}")
    print(f"  Is Watertight: {info.is_watertight}")
    print(f"  Self-intersecting: {info.is_intersecting}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", help="Path to the mesh file")
    args = parser.parse_args()
    benchmark(args.mesh)
