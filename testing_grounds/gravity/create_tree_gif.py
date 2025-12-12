#!/usr/bin/env python
"""
Script to create an orbiting animation of the 3D tree structure and save as GIF.
Run this from the same directory as tree_building.ipynb after building the tree.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add parent directories to path for gravhydro import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from gravhydro.gravity.tree import Tree

def get_cube_edges_mpl(node):
    """Get cube edges for matplotlib plotting"""
    x = [node.xmin, node.xmax]
    y = [node.ymin, node.ymax]
    z = [node.zmin, node.zmax]
    
    edges = [
        # Bottom face
        [(x[0],y[0],z[0]), (x[1],y[0],z[0])],
        [(x[1],y[0],z[0]), (x[1],y[1],z[0])],
        [(x[1],y[1],z[0]), (x[0],y[1],z[0])],
        [(x[0],y[1],z[0]), (x[0],y[0],z[0])],
        # Top face
        [(x[0],y[0],z[1]), (x[1],y[0],z[1])],
        [(x[1],y[0],z[1]), (x[1],y[1],z[1])],
        [(x[1],y[1],z[1]), (x[0],y[1],z[1])],
        [(x[0],y[1],z[1]), (x[0],y[0],z[1])],
        # Vertical edges
        [(x[0],y[0],z[0]), (x[0],y[0],z[1])],
        [(x[1],y[0],z[0]), (x[1],y[0],z[1])],
        [(x[1],y[1],z[0]), (x[1],y[1],z[1])],
        [(x[0],y[1],z[0]), (x[0],y[1],z[1])]
    ]
    return edges


def create_tree_orbit_gif(tree, data, output_path='tree_3d_orbit.gif', 
                          n_frames=15*6, elevation=20, fps=15):
    """
    Create an orbiting animation of a 3D tree structure.
    
    Parameters
    ----------
    tree : Tree
        The built tree structure
    data : ndarray
        Particle positions (N, 3)
    output_path : str
        Path to save the GIF
    n_frames : int
        Number of frames for one full rotation
    elevation : float
        Fixed elevation angle for the camera
    fps : int
        Frames per second for the GIF
    """
    # Collect all edges from leaf nodes
    all_edges = []
    nodes_to_check = [tree.root]
    while nodes_to_check:
        node = nodes_to_check.pop()
        if len(node.particle_set) == 1:  # Leaf condition
            all_edges.extend(get_cube_edges_mpl(node))
        else:
            nodes_to_check.extend(node.children)
    
    print(f"Found {len(all_edges)//12} leaf nodes to plot")
    
    # Create frames for the GIF
    frames = []
    
    print(f"Generating {n_frames} frames for orbiting animation...")
    
    for i, azim in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot particles
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                   c='k', s=50, alpha=0.8, label='Particles')
        
        # Plot tree edges
        for edge in all_edges:
            xs = [edge[0][0], edge[1][0]]
            ys = [edge[0][1], edge[1][1]]
            zs = [edge[0][2], edge[1][2]]
            ax.plot(xs, ys, zs, 'teal', linewidth=0.8, alpha=0.6)
        
        # Set viewing angle
        ax.view_init(elev=elevation, azim=azim)
        
        # Hide axes
        ax.set_axis_off()
        
        # Set consistent axis limits
        ax.set_xlim(tree.root.xmin, tree.root.xmax)
        ax.set_ylim(tree.root.ymin, tree.root.ymax)
        ax.set_zlim(tree.root.zmin, tree.root.zmax)
        
        # Capture frame to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        frames.append(img)
        
        plt.close(fig)
        buf.close()
        
        if (i + 1) % 10 == 0:
            print(f"  Frame {i + 1}/{n_frames} complete")
    
    # Save as GIF using PIL
    print(f"Saving GIF to {output_path}...")
    duration = int(1000 / fps)  # Convert fps to ms per frame
    frames[0].save(output_path, save_all=True, append_images=frames[1:], 
                   duration=duration, loop=0)
    print(f"Done! GIF saved as '{output_path}'")
    
    return frames


if __name__ == '__main__':
    # Generate sample data if run directly
    np.random.seed(41)
    data = np.random.uniform(-1, 1, size=(30, 3))
    masses = np.random.uniform(1, 2, size=(data.shape[0])) * 10
    
    # Build tree
    tree = Tree()
    tree.build(data, masses)
    
    # Create GIF
    frames = create_tree_orbit_gif(tree, data)
    
    # Show preview
    plt.figure(figsize=(8, 8))
    plt.imshow(frames[0])
    plt.axis('off')
    plt.title('Preview (first frame)')
    plt.show()
