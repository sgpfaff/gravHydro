"""
Script to create an orbiting GIF of the 3D tree structure.
Run this after building the tree in the notebook.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import gravhydro as gh
from gravhydro import Tree

# Generate the same data as in the notebook (3D example)
np.random.seed(41)
data = np.random.uniform(-1, 1, size=(30, 3))
masses = np.random.uniform(1, 2, size=(data.shape[0])) * 10

# Build the tree
tree = Tree()
tree.build(data, masses)

# Function to create cube edges for a node
def get_cube_edges_mpl(node):
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

# Collect all cube edges from the tree (same logic as tree.plot with style='leaf')
all_edges = []
nodes_to_check = [tree.root]
while nodes_to_check:
    node = nodes_to_check.pop()
    condition = (len(node.particle_set) == 1)  # leaf style
    if condition:
        all_edges.extend(get_cube_edges_mpl(node))
    else:
        nodes_to_check.extend(node.children)

print(f"Collected {len(all_edges)} edges from {len(data)} particles")

# Create figure
fig = plt.figure(figsize=(8, 8), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    
    # Plot particles
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
               s=masses*3, c='teal', alpha=0.8, edgecolors='none', linewidth=0.5)
    
    # Plot tree edges (leaf nodes only)
    for edge in all_edges:
        ax.plot3D([edge[0][0], edge[1][0]], 
                  [edge[0][1], edge[1][1]], 
                  [edge[0][2], edge[1][2]], 
                  'k-', alpha=0.5, linewidth=0.8)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title('3D Tree Structure (Orbiting View)')
    
    # Set consistent axis limits based on root node
    ax.set_xlim(tree.root.xmin, tree.root.xmax)
    ax.set_ylim(tree.root.ymin, tree.root.ymax)
    ax.set_zlim(tree.root.zmin, tree.root.zmax)
    
    # Orbit the camera - full 360 degree rotation
    ax.view_init(elev=25, azim=frame * 3)  # 120 frames * 3 degrees = 360 degrees
    
    return ax,

# Create animation with 120 frames for a full orbit
n_frames = 120
print(f"Creating animation with {n_frames} frames...")
anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)

# Save as GIF
gif_path = 'tree_orbit.gif'
print(f"Saving GIF to {gif_path}...")
anim.save(gif_path, writer=PillowWriter(fps=24))
print(f"GIF saved successfully to: {gif_path}")

plt.close()
print("Done!")
