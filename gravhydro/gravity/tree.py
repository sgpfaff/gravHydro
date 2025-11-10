
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
        root_lims = [np.min(qs)*1.1, np.max(qs)*1.1]
        self.root = Node(root_lims, root_lims, root_lims) # create root node
        self.positions = qs
        self.masses = masses
        for i in range(len(qs)):
            self._insertParticle(self.root, i)
        self._calculateMass()
        self._calculateCOM()

    def force(self, q, threshold):
        def _belowThreshold(node, q, threshold):
            '''check if node is sufficiently far away from point q'''
            dist = np.linalg.norm(node.center_of_mass - q, axis=1)
            size = node.size
            return size / dist < threshold
        def _nodeForces(q, node):
            if len(node.particle_inds) == 0:
                return
            threshold_mask = _belowThreshold(node, q, threshold)
            if np.any(~threshold_mask):
                # calculate forces for points that are below threshold
                dist = np.linalg.norm(node.center_of_mass - q[~threshold_mask], axis=1)
                forces[~threshold_mask] = forces[~threshold_mask] + (node.mass / dist[:,None]**2 * (node.center_of_mass - q[~threshold_mask]) / dist[:,None])
                for child in node.children: # calculate forces for remaining points above threshold
                    _nodeForces(q[threshold_mask], child)
        forces = np.zeros_like(q)
        _nodeForces(q, self.root)
        return forces
   
    def _insertParticle(self, node, particle_ind):
        '''
        Insert a particle into the tree node.

        Parameters
        ----------
        node : Node
            Current node in the tree.
        particle_ind : array_like
            Index of the particle to insert.
        '''
        if not node.inside(self.positions[particle_ind]): # check if particle is inside node. Return if it is not
            return
        else: # when the particle is inside the node
            node.add_particle(particle_ind) # add particle to this node
            if len(node.particle_inds) == 1: # if the node has no particles (leaf node)
                return # then we are done
            else: # otherwise,
                if len(node.children) > 0:
                    octant = node.getOctant(self.positions[particle_ind]) # find correct octant for this particle
                    self._insertParticle(node.children[octant], particle_ind)
                    # .add_particle(particle_ind)
                    
                else: # when the current node isn't already divided
                    node.split() # split the node into children.
                    for ind in node.particle_inds: # and redistribute the existing particles among the children
                        octant = node.getOctant(self.positions[ind]) # find correct octant for this particle
                        self._insertParticle(node.children[octant], ind) # This is the issue...

    def _calculateCOM(self):
        '''Calculate the COM of branch nodes recursively.'''
        def _calculateCOMnode(node):
            if node.leaf:
                node.center_of_mass = self.positions[node.particle_inds]
            else:
                xcom = np.sum(self.positions[...,0][node.particle_inds] * self.masses[node.particle_inds]) / node.mass
                ycom = np.sum(self.positions[...,1][node.particle_inds] * self.masses[node.particle_inds]) / node.mass
                zcom = np.sum(self.positions[...,2][node.particle_inds] * self.masses[node.particle_inds]) / node.mass
                node.center_of_mass = np.array([xcom, ycom, zcom])
                for child in node.children:
                    _calculateCOMnode(child)
        _calculateCOMnode(self.root)
        
        
    def _calculateMass(self):
        '''Calculate the mass of branch nodes recursively.'''
        def _calculateMassnode(node):
            if node.leaf:
                node.mass = self.masses[node.particle_inds]
            else:
                node.mass = np.sum(self.masses[node.particle_inds])
                for child in node.children:
                    _calculateMassnode(child)
        _calculateMassnode(self.root)

    def forces(self, q, p):
        # Placeholder for actual force computation
        return np.zeros_like(q)
    
    def plot(self, projection='3D', data=None):
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

            def plot_tree(tree, data):
                plt.figure(figsize=(6,6))
                
                nodes_to_plot = [tree.root]
                while nodes_to_plot:
                    node = nodes_to_plot.pop()

                    nodes_to_plot.extend(node.children)
                    for child in node.children:
                        if len(child.particle_inds) == 1:
                            plot_node(child)
                if np.all(data != None):
                    plt.scatter(data[:,0], data[:,1], s=20)
                plt.axis('equal')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()
            plot_tree(self, data)
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
            nodes_to_plot = []
            while nodes_to_check:
                node = nodes_to_check.pop()
                nodes_to_check.extend(node.children)
                for child in node.children:
                    if len(child.particle_inds) == 1:
                        nodes_to_plot.append(child)
                        edges_x, edges_y, edges_z = get_cube_edges(child)
                        all_edges_x.extend(edges_x)
                        all_edges_y.extend(edges_y)
                        all_edges_z.extend(edges_z)
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
        self.size = self.xmax - self.xmin # size of node (assuming cubic)

        self.particle_inds = [] # particle indices

        self.children = [] # lower-right-front, lower-right-back, lower-left-front, lower-left-back, upper-right-front, upper-right-back, upper-left-front, upper-left-back

    def add_particle(self, ind):
        '''
        Add a particle to this node.

        Parameters
        ----------
        q : array_like
            Position of the particle to add.
        p : array_like
            Momentum of the particle to add.
        '''
        if len(self.particle_inds) == 0: # we are inserting the first particle now
            self.leaf = True
        else:
            self.leaf = False
        self.particle_inds.append(ind)

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
    # def center_of_mass(self):
    #     '''Center of mass of the node.'''
    #     if self.leaf:
    #         pass
    #     if self._center_of_mass is not None:
    #         return self._center_of_mass
    #     else:
    #         self._calculateCOM()
    #         return self._center_of_mass
        
    # def mass(self):
    #     '''Mass of the node.'''
    #     if self._mass is not None:
    #         return self._mass
    #     else:
    #         self._calculateMass()
    #         return self._mass

    # def _calculateCOM(self):
    #     '''Calculate the COM of nody.'''
    #     if self.leaf:
    #             pass
               
    #     else:
    #         total_mass = np.sum(self.masses[self.particle_inds])
    #         for child in self.children:
    #             child.calculateCOM()
    #             total_mass += child.mass
    #             com_sum += child.mass * child.com
    #         self.mass = total_mass
    #         if total_mass > 0:
    #             self._center_of_mass = com_sum / total_mass
    #         else:
    #             self._center_of_mass = np.array([0.0, 0.0, 0.0])

    # def _calculateMass(self):
    #     '''Calculate the mass of node.'''
    #     self._mass = np.sum(self.masses[self.particle_inds])
    
