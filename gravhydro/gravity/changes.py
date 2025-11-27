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
        mins = np.min(qs, axis=0) * 1.1
        maxs = np.max(qs, axis=0) * 1.1
        # Make it cubic (required for octree)
        size = max(maxs - mins)
        center = (mins + maxs) / 2
        half = size / 2 * 1.01  # Small padding
        root_lims = (center[0] - half, center[0] + half)
        root_ylims = (center[1] - half, center[1] + half)
        root_zlims = (center[2] - half, center[2] + half)
        
        self.root = Node(root_lims, root_ylims, root_zlims)
        self.positions = qs
        self.indices = np.arange(len(qs))
        self.masses = masses
        self.N = len(qs)
        
        for i in range(self.N):
            self._insertParticle(self.root, i)
        self._calculateMass()
        self._calculateCOM()

    def force(self, threshold):
        """
        Compute gravitational forces on all particles using Barnes-Hut algorithm.
        
        For each particle, walk the tree and sum forces from all other particles/nodes.
        Uses the opening angle criterion: if size/distance < threshold, use COM approximation.
        """
        forces = np.zeros_like(self.positions)
        positions = self.positions  # Local reference for speed
        masses = self.masses
        threshold_sq = threshold * threshold  # Compare squared values to avoid sqrt
        
        def _computeForceOnParticle(particle_idx, node, force_accum):
            """Compute force on particle from this node (and its children if needed)."""
            # Skip empty nodes
            n_particles = len(node.particle_inds)
            if n_particles == 0:
                return
            
            # If this is a leaf node with only this particle, skip (no self-force)
            if node.leaf and n_particles == 1 and node.particle_inds[0] == particle_idx:
                return
            
            # Get particle position (use local ref)
            q_particle = positions[particle_idx]
            
            # Compute distance to node's center of mass
            r_vec = node.center_of_mass - q_particle
            r_sq = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]
            
            # Avoid self-interaction (when particle is at the COM)
            if r_sq < 1e-20:
                for child in node.children:
                    _computeForceOnParticle(particle_idx, child, force_accum)
                return
            
            # Check opening angle criterion: (size/r)^2 < threshold^2
            size_sq = node.size * node.size
            if node.leaf or (size_sq / r_sq < threshold_sq):
                # Use COM approximation for this node
                # But we need to exclude the particle itself if it's in this node
                if node.particle_set is not None and particle_idx in node.particle_set:
                    # Can't use COM - particle is inside this node, must recurse
                    for child in node.children:
                        _computeForceOnParticle(particle_idx, child, force_accum)
                    return
                else:
                    # Safe to use COM approximation
                    r_mag = np.sqrt(r_sq)
                    factor = G * node.mass * masses[particle_idx] / (r_sq * r_mag)
                    force_accum[0] += factor * r_vec[0]
                    force_accum[1] += factor * r_vec[1]
                    force_accum[2] += factor * r_vec[2]
                    return
            else:
                # Node is too close, recurse into children
                for child in node.children:
                    _computeForceOnParticle(particle_idx, child, force_accum)
                return
        
        # Compute force on each particle
        for i in range(self.N):
            force_accum = forces[i]  # Direct reference to output array
            _computeForceOnParticle(i, self.root, force_accum)
        
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
            if len(node.particle_inds) == 0:
                # Empty node - set COM to center of node bounds
                node.center_of_mass = np.array([
                    (node.xmin + node.xmax) / 2,
                    (node.ymin + node.ymax) / 2,
                    (node.zmin + node.zmax) / 2
                ])
            elif node.leaf:
                # For leaf nodes with single particle, flatten to 1D array
                node.center_of_mass = self.positions[node.particle_inds[0]]
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
            if len(node.particle_inds) == 0:
                # Empty node
                node.mass = 0.0
            elif node.leaf:
                # For leaf nodes, extract scalar mass
                node.mass = self.masses[node.particle_inds[0]]
            else:
                node.mass = np.sum(self.masses[node.particle_inds])
                for child in node.children:
                    _calculateMassnode(child)
        _calculateMassnode(self.root)
