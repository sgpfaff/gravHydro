from .gravity.tree import Tree
from .integration import leapfrogStep
import numpy as np

class Simulation:
    def __init__(self):
        # self.xmin,  self.xmax= None, None
        # self.ymin,  self.ymax= None, None
        # self.zmin,  self.zmax= None, None
        self.tree = None

    def run(self, q0, p0, masses, ts, threshold=0.5):
        '''
        Run the simulation. Integrates particles forward in time, 
        subject to gravitational and hydrodynamic forces.

        Parameters
        ----------
        q0 : array_like
            Initial positions of particles.
        p0 : array_like
            Initial momenta of particles.
        ts : float
            Times to return the position and momentum for the particles.

        Returns
        -------
        qs : array_like
            Positions of particles at each time in ts.
        ps : array_like
            Momenta of particles at each time in ts.
        '''

        ### Create the tree structure for gravitational calculations ###
        self.tree = Tree() # initialize tree with these bounds
        self.tree.build(q0, masses) # build the initial tree

        ### Time integration loop ###
        qs, ps = np.zeros((*ts.shape, *q0.shape)), np.zeros((*ts.shape, *q0.shape)) # initialize arrays to hold positions and momenta
        qs[0], ps[0] = q0, p0 # set initial conditions
        dt = ts[1] - ts[0] # calculate timestep
        for i, _ in enumerate(ts[1:], start=1): # loop over time steps
            self.tree.build(qs[i-1], masses) # rebuild tree at each step
            force = self._gravityForces(qs[i-1], threshold=threshold) # evaluate the total force on all particles
            q, p = leapfrogStep(qs[i-1], ps[i-1], dt, force) # integrate forward one timestep
            qs[i], ps[i] = q, p # append new positions and momenta
        return qs, ps

    def _evaluateTotalForce(self, q, p):
        '''
        Evaluate gravitational and hydrodynamic forces on particles.

        Parameters
        ----------
        q : array_like
            Positions of particles.
        p : array_like
            Momenta of particles.

        Returns
        -------
        forces : array_like
            Computed forces on particles.
        '''
        return self._gravityForces(q, p) + self._hydroForces(q, p)
    
    def _gravityForces(self, q, threshold=0.5):
        '''
        Compute gravitational forces using the tree structure.

        Parameters
        ----------
        q : array_like
            Positions of particles.
        p : array_like
            Momenta of particles.

        Returns
        -------
        gravity_forces : array_like
            Gravitational forces on particles.
        '''
        # Placeholder for actual tree-based gravity force computation
        return self.tree.force(q, threshold)

    def _hydroForces(self, q, p):
        '''
        Compute hydrodynamic forces on particles.

        Parameters
        ----------
        q : array_like
            Positions of particles.
        p : array_like
            Momenta of particles.

        Returns
        -------
        hydro_forces : array_like
            Hydrodynamic forces on particles.
        '''
        # Placeholder for actual hydrodynamic force computation
        hydro_forces = None
        return hydro_forces
    
