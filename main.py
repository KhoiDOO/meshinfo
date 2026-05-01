import argparse

from viewer.meshviewer import MeshViewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", action="store_true", help="Check connected components and related properties (default: False)")
    parser.add_argument("--intersect", action="store_true", help="Check for intersecting faces (default: False)")
    parser.add_argument("--nonmanifold", action="store_true", help="Check for non-manifold vertices (default: False)")
    parser.add_argument("--geometry", action="store_true", help="Check geometric properties (default: False)")
    parser.add_argument("--topology", action="store_true", help="Check topology properties (default: False)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    app = MeshViewer(
        check_components=args.components,
        check_intersection=args.intersect,
        check_nonmanifold_vertices=args.nonmanifold,
        check_geometry=args.geometry,
        check_topology=args.topology,
        verbose=args.verbose
    )
    app.run()
