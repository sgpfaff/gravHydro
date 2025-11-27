
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
G = 1.0

# Pre-allocated zero vector for performance
_ZERO_VEC = np.zeros(3)

class Tree:
    def __init__(self):
        self.root = None
        
    def build(self, qs, masses):
        '''
        Build the tree structure from particle positions.

        Parameters
        ----------
        qs : array_like
            Positions of particles. Shape (N, 3) where N is number of particles.
        '''
        ### Limits of root node ###
        # Use separate min/max for each axis for non-cubic domains
        mins = np.min(qs, axis=0) 
        maxs = np.max(qs, axis=0) #* 1.1
        # Make it cubic (required for octree)
        size = max(maxs - mins)
        center = (mins + maxs) / 2
        half = size / 2 * 1.01  # Small padding
        root_lims = (center[0] - half, center[0] + half)
        root_ylims = (center[1] - half, center[1] + half)
        root_zlims = (center[2] - half, center[2] + half)
        
        self.root = Node(root_lims, root_ylims, root_zlims)
        self.positions = qs
        self._ZERO_VEC = np.zeros_like(self.positions)
        self.indices = np.arange(len(qs))
        self.masses = masses
        N = len(qs)
        self.N = N

        for i, q in enumerate(qs):
            self._insertParticle(self.root, q, i)

    def force(self, threshold):
        """
        Compute gravitational forces on all particles using Barnes-Hut algorithm.
        
        For each particle, walk the tree and sum forces from all other particles/nodes.
        Uses the opening angle criterion: if size/distance < threshold, use COM approximation.
        """
        n = self.N
        forces = np.zeros((n, 3))  #np.zeros_like(self.positions)
        positions = self.positions  # Local reference for speed
        masses = self.masses
        threshold2 = threshold * threshold  # Compare squared values to avoid sqrt

        for i in range(self.N): # O(N)
            stack = [self.root]
            while stack:
                node = stack.pop()
                pt_set = node.particle_set
                n_pts = len(pt_set)
                if n_pts == 0: # skip if node is empty
                    continue
                elif i in pt_set: # open node if it contains this pt
                    stack.extend(node.children)
                # elif n_pts == 1 and node.leaf and i in pt_set: # skip if the node is a leaf and contains only the current particle
                #     return
                else: # check if node meets opening criteria
                    q_i = positions[i]
                    if type(node.center_of_mass) == type(None):
                        self._calculateNodeCOMandMass(node) # calculate node masses on the fly. Updates Node COM attr
                    r_vec = node.center_of_mass - q_i
                    r2 = np.dot(r_vec,r_vec.T)#np.dot(r_vec, r_vec)
                    if node.leaf or node.size2/r2 < threshold2: # if pt is far enough away to calcaulate the force or leaf node 
                        forces_i = forces[i] # create reference for forces on pt i
                        m_i = masses[i]
                        prefactor = G * m_i * node.mass / r2**(3/2)
                        forces_i += prefactor * r_vec
                    else: # otherwise, open the node
                        stack.extend(node.children)
        return forces
   
    def _insertParticle(self, node, q, i):
        '''
        Insert a particle into the tree node.

        Parameters
        ----------
        node : Node
            Current node in the tree.
        particle_ind : array_like
            Index of the particle to insert.
        '''
        if not node.inside(q): # check if particle is inside node. Return if it is not
            return
        else: # when the particle is inside the node
            node.add_particle(i) # add particle to this node
            if len(node.particle_set) == 1: # if the node has no particles (leaf node)
                return # then we are done
            else: # otherwise,
                if len(node.children) > 0:
                    octant = node.getOctant(q) # find correct octant for this particle
                    self._insertParticle(node.children[octant], q, i)
                    # .add_particle(particle_ind)
                else: # when the current node isn't already divide
                    node.split() # split the node into children.
                    for i in node.particle_set: # and redistribute the existing particles among the children
                        q_i = self.positions[i]
                        octant = node.getOctant(q_i) # find correct octant for this particle
                        self._insertParticle(node.children[octant], q_i, i)
    def _calculateNodeCOMandMass(self, node):
        particle_inds = None
        if len(node.particle_set) == 0:
            # Empty node - set COM to center of node bounds
            node.center_of_mass = np.array([
                (node.xmin + node.xmax) / 2,
                (node.ymin + node.ymax) / 2,
                (node.zmin + node.zmax) / 2
            ])
            if node.mass == None:
                node.mass = 0
        elif len(node.particle_set) == 1:
            # Save particle mass and position if only 1 particle
            if node.mass == None:
                particle_inds = list(node.particle_set)
                node.mass = self.masses[particle_inds]
            node.center_of_mass = self.positions[particle_inds[0] if particle_inds != None else list(node.particle_set)[0]]
        else:
            particle_inds = list(node.particle_set)
            qs = self.positions[particle_inds]
            ms = self.masses[particle_inds]
            if node.mass == None:
                node.mass = np.sum(self.masses[particle_inds])
            totalMass = node.mass
            xCOM = np.sum(qs[..., 0] * ms / totalMass)#np.sum(self.positions[...,0][particle_inds] * self.masses[particle_inds]) / node.mass
            yCOM = np.sum(qs[..., 1] * ms / totalMass)#np.sum(self.positions[...,1][particle_inds] * self.masses[particle_inds]) / node.mass
            zCOM = np.sum(qs[..., 2] * ms / totalMass)#np.sum(self.positions[...,2][particle_inds] * self.masses[particle_inds]) / node.mass
            node.center_of_mass = np.array([xCOM, yCOM, zCOM])
    
    def plot(self, projection='3D', data=None, style='leaf'):
        '''
        Plot the tree structure and optionally particle positions.

        Parameters
        ----------
        q : array_like, optional
            Positions of particles to plot.
        '''
        if projection == '2D':
            def plot_node(node):
                x_coords = [node.xmin, node.xmax, node.xmax, node.xmin, node.xmin]
                y_coords = [node.ymin, node.ymin, node.ymax, node.ymax, node.ymin]
                plt.plot(x_coords, y_coords, 'r-', lw=1, alpha=0.5)

            def plot_tree(tree, data, style):
                plt.figure(figsize=(6,6))
                nodes_to_plot = [tree.root]
                while nodes_to_plot:
                    node = nodes_to_plot.pop()
                    if style == 'leaf':
                        condition = (len(node.particle_set) == 1)#node.leaf
                    # else: 
                    #     condition = (len(node.particle_set) == 1)
                    if condition:
                        plot_node(node)
                    else:
                        nodes_to_plot.extend(node.children)
                    # nodes_to_plot.extend(node.children)
                    # for child in node.children:
                    #     if len(child.particle_set) == 1:
                    #         plot_node(child)
                if np.all(data != None):
                    plt.scatter(data[:,0], data[:,1], s=20)
                plt.axis('equal')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()
            plot_tree(self, data, style)

        elif projection == '3D':
            if np.all(data != None):
                # Create traces for particles
                particle_trace = go.Scatter3d(
                    x=data[:,0], 
                    y=data[:,1], 
                    z=data[:,2],
                    mode='markers',
                    marker=dict(size=5, color='blue'),
                    name='Particles'
                )
            else:
                particle_trace = go.Scatter3d(
                    x=[], 
                    y=[], 
                    z=[],
                    mode='markers',
                    marker=dict(size=5, color='blue'),
                    name='Particles'
                )

            # Function to create cube edges for a node
            def get_cube_edges(node):
                x = [node.xmin, node.xmax]
                y = [node.ymin, node.ymax]
                z = [node.zmin, node.zmax]
                
                edges_x, edges_y, edges_z = [], [], []
                
                # Create all 12 edges of the cube
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
                
                for edge in edges:
                    edges_x.extend([edge[0][0], edge[1][0], None])
                    edges_y.extend([edge[0][1], edge[1][1], None])
                    edges_z.extend([edge[0][2], edge[1][2], None])
                
                return edges_x, edges_y, edges_z

            # Collect all cube edges
            all_edges_x, all_edges_y, all_edges_z = [], [], []
            nodes_to_check = [self.root]
            while nodes_to_check:
                node = nodes_to_check.pop()
                if style=='leaf':
                    condition = (len(node.particle_set) == 1)#node.leaf
                if condition:
                    edges_x, edges_y, edges_z = get_cube_edges(node)
                    all_edges_x.extend(edges_x)
                    all_edges_y.extend(edges_y)
                    all_edges_z.extend(edges_z)
                else:
                    nodes_to_check.extend(node.children)
                        
            # Create trace for tree structure
            tree_trace = go.Scatter3d(
                x=all_edges_x,
                y=all_edges_y,
                z=all_edges_z,
                mode='lines',
                line=dict(color='red', width=2),
                name='Tree Structure'
            )

            # Create figure
            fig = go.Figure(data=[particle_trace, tree_trace])
            fig.update_layout(
                title='3D Tree Visualization',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                width=800,
                height=800
            )
            fig.show()
        pass


class Node:
    def __init__(self, xlims, ylims, zlims):
        self.xmin, self.xmax = xlims
        self.ymin, self.ymax = ylims
        self.zmin, self.zmax = zlims
        self.leaf = True
        self.center_of_mass = None # center of mass
        self.mass = None # total mass
        size = self.xmax - self.xmin # size of node (assuming cubic)
        self.size2 = size**2
        #self.particle_inds = [] # particle indices
        self.particle_set = set()
        self.children = [] # lower-right-front, lower-right-back, lower-left-front, lower-left-back, upper-right-front, upper-right-back, upper-left-front, upper-left-back

    def add_particle(self, i):
        '''
        Add a particle to this node.

        Parameters
        ----------
        q : array_like
            Position of the particle to add.
        p : array_like
            Momentum of the particle to add.
        '''
        if len(self.particle_set) == 0: # we are inserting the first particle now
            self.leaf = True
        else:
            self.leaf = False
        self.particle_set.add(i)

    def inside(self, q):
        '''
        Check if a point is inside the node's boundaries.

        Parameters
        ----------
        q : array_like
            Position of the point to check.
        
        Returns
        -------
        bool
            True if the point is inside the node, False otherwise.
        '''
        return (self.xmin <= q[0] <= self.xmax and
                self.ymin <= q[1] <= self.ymax and
                self.zmin <= q[2] <= self.zmax)
    
    def getOctant(self, q):
        '''
        Determine which octant the point belongs to.

        Parameters
        ----------
        q : array_like
            Position of the point to check.
        
        Returns
        -------
        int
            Index of the octant (0-7).
        '''
        xmid = (self.xmin + self.xmax) / 2
        ymid = (self.ymin + self.ymax) / 2
        zmid = (self.zmin + self.zmax) / 2

        if q[0] < xmid:
            if q[1] < ymid:
                if q[2] < zmid:
                    return 0
                else:
                    return 4
            else:
                if q[2] < zmid:
                    return 2
                else:
                    return 6
        else:
            if q[1] < ymid:
                if q[2] < zmid:
                    return 1
                else:
                    return 5
            else:
                if q[2] < zmid:
                    return 3
                else:
                    return 7
                

    def split(self):
        '''
        Split the node into 8 children.

        Parameters
        ----------
        q : array_like
            Position of the particle that caused the split.
        '''
        xmid = (self.xmin + self.xmax) / 2
        ymid = (self.ymin + self.ymax) / 2
        zmid = (self.zmin + self.zmax) / 2

        # Create 8 children nodes
        self.children = [
            Node((self.xmin, xmid), (self.ymin, ymid), (self.zmin, zmid)),
            Node((xmid, self.xmax), (self.ymin, ymid), (self.zmin, zmid)),
            Node((self.xmin, xmid), (ymid, self.ymax), (self.zmin, zmid)),
            Node((xmid, self.xmax), (ymid, self.ymax), (self.zmin, zmid)),
            Node((self.xmin, xmid), (self.ymin, ymid), (zmid, self.zmax)),
            Node((xmid, self.xmax), (self.ymin, ymid), (zmid, self.zmax)),
            Node((self.xmin, xmid), (ymid, self.ymax), (zmid, self.zmax)),
            Node((xmid, self.xmax), (ymid, self.ymax), (zmid, self.zmax)),
        ]
        self.leaf = False


    def center_of_mass(self):
        '''Center of mass of the node.'''
        if len(self.particle_set) == 0:
            self.center_of_mass = np.array([
                (self.xmin + self.xmax) / 2,
                (self.ymin + self.ymax) / 2,
                (self.zmin + self.zmax) / 2])
        if self._center_of_mass is not None:
            return self._center_of_mass
        else:
            self._calculateCOM()
            return self._center_of_mass
        
    def mass(self, masses):
        '''Mass of the node.'''
        if self._mass is not None:
            return self._mass
        else:
            self._calculateMass()
            return self._mass