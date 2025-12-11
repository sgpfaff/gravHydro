from .gravity.tree import Tree
from .gravity.numba_gravity import (
    NumbaGravity, 
    direct_force_summation_numba, 
    is_numba_available,
    NUMBA_AVAILABLE
)
from .hydro import pressureAcc, densities
from .hydro.numba_hydro import NumbaHydro, NUMBA_AVAILABLE as NUMBA_HYDRO_AVAILABLE
from .utils import convert_to_internal, convert_to_physical, astropy_to_pynbody
from .integration import leapfrogStep
import numpy as np
import pynbody as pyn
import astropy.units as u
from tqdm import tqdm


G = 1.0  # Gravitational constant

class Simulation:
    def __init__(self, use_numba=None):
        """
        Initialize the simulation.
        
        Parameters
        ----------
        use_numba : bool, optional
            Whether to use Numba acceleration for gravity calculations.
            Default is None, which uses Numba if available.
        """
        self.tree = None
        self.time = np.array([])*u.yr  # Initialize time as a NumPy array
        self.nParticles = 0
        self.positions = None
        self.velocities = None
        self.masses = None
        self._numba_gravity = None
        self._numba_hydro = None
        
        # Set up Numba usage
        if use_numba is None:
            self.use_numba = NUMBA_AVAILABLE
        else:
            self.use_numba = use_numba and NUMBA_AVAILABLE
        
        if self.use_numba:
            self._numba_gravity = NumbaGravity(use_numba=True)
            if NUMBA_HYDRO_AVAILABLE:
                self._numba_hydro = NumbaHydro(use_numba=True)

    def run(
            self, 
            positions, 
            velocities, 
            masses, 
            ts, 
            threshold=0.5, 
            gravityOnly = False,
            gravityMethod='directSummation', 
            use_numba=True, 
            softening=None,
            returnAstropy=True,
            hydroOnly = False,
            h = None, 
            rho0 = None,
            cs = None,
            gamma = 5/3,
            unitLength = None,
            unitTime = None,
            unitMass = None
            ):
        '''
        Run the simulation. Integrates particles forward in time, 
        subject to gravitational and hydrodynamic forces.

        Parameters
        ----------
        positions : array_like
            Initial positions of particles.
        velocities : array_like
            Initial velocities of particles.
        ts : float
            Times to return the position and momentum for the particles.
        threshold : float, optional
            Opening angle for tree method. Default is 0.5.
        gravityMethod : str, optional
            Method for gravity calculation: 'tree', 'directSummation', or 'numba'.
            Default is 'tree'.
        use_numba : bool, optional
            Override the simulation's Numba setting for this run.
            If None, uses the simulation's default setting.
        return_physical : bool, optional
            Whether to return astropy quantities.

        Returns
        -------
        qs : array_like
            Positions of particles at each time in ts.
        ps : array_like
            Momenta of particles at each time in ts.
        '''
        # Determine whether to use Numba for this run
        if use_numba is None:
            run_use_numba = self.use_numba
        else:
            run_use_numba = use_numba and NUMBA_AVAILABLE

        ### Create the tree structure for gravitational calculations ###
        
        ### Time integration loop ###
        self.time = np.append(self.time, ts)
        self.masses = masses
        self.nParticles = len(masses)
        if softening != None:
            softening = convert_to_internal(softening, unitLength, unitTime, unitMass)
        else:
            softening = 1e-10
        if h != None:
            h = convert_to_internal(h, unitLength, unitTime, unitMass)
        q0, v0, masses, ts = convert_to_internal(positions, unitLength, unitTime, unitMass), convert_to_internal(velocities, unitLength, unitTime, unitMass), convert_to_internal(masses, unitLength, unitTime, unitMass), convert_to_internal(ts, unitLength, unitTime, unitMass)   
        
        # Track original dimensionality for 2D simulations
        original_ndim = q0.shape[1]
        
        if rho0 != None:
            rho0 = convert_to_internal(rho0, unitLength, unitTime, unitMass)
        if cs != None:
            cs = convert_to_internal(cs, unitLength, unitTime, unitMass)
        qs, ps = np.zeros((*ts.shape, *q0.shape)), np.zeros((*ts.shape, *v0.shape)) # initialize arrays to hold positions and momenta
        p0 = v0 * masses[:, None]
        qs[0], ps[0] = q0, p0 # set initial conditions
        dt = ts[1] - ts[0] # calculate timestep

        if not hydroOnly:
            if gravityMethod == 'tree':
                # initialize tree with these bounds
                tree = Tree()
                tree.build(q0, masses) # build the initial tree
            else:
                tree = None
        if not gravityOnly:
            if rho0 == None:
                rho0 = np.mean(densities(q0, masses, h))


        for i, _ in enumerate(tqdm(ts[1:]), start=1): # loop over time steps
            q_half = qs[i-1] + (ps[i-1]/masses[:, np.newaxis]) * (dt/2) # half-step position update
            if not hydroOnly:
                # Gravity requires 3D - pad 2D positions if necessary
                if original_ndim == 2:
                    q_half_3d = np.column_stack([q_half, np.zeros(len(q_half))])
                else:
                    q_half_3d = q_half
                    
                if gravityMethod == 'tree':
                    tree = Tree()
                    tree.build(q_half_3d, masses) # rebuild tree at HALF-STEP positions
                else:
                    tree = None
                gravityForce = self._gravityForces(q_half_3d, masses, threshold=threshold, method=gravityMethod, tree=tree, use_numba=run_use_numba, softening=softening) # evaluate the total force on all particles
                
                # Slice gravity force back to original dimensions
                if original_ndim == 2:
                    gravityForce = gravityForce[:, :2]
                
                if gravityOnly:
                    force = gravityForce
                else:
                    hydroForce = self._hydroForces(q_half, masses, h, rho0, cs, gamma)
                    force = gravityForce + hydroForce
            else:
                hydroForce = self._hydroForces(q_half, masses, h, rho0, cs, gamma)
                force = hydroForce
            #print(force)
            ps[i] = ps[i-1] + (force * dt) # full-step momentum update
            qs[i] = q_half + (ps[i]/masses[:, np.newaxis]) * (dt/2) # full-step position update
            vs = ps / masses[:,None]
        qs, vs = convert_to_physical(qs, 'length', returnAstropy, unitLength, unitTime, unitMass), convert_to_physical(vs, 'speed', returnAstropy, unitLength, unitTime, unitMass)
        self.positions, self.velocities = qs, vs

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
    
    def _gravityForces(self, q, masses, threshold, tree=None, method='tree', use_numba=False, softening=None):
        '''
        Compute gravitational forces using the tree structure.

        Parameters
        ----------
        q : array_like
            Positions of particles.
        masses : array_like
            Particle masses.
        threshold : float
            Opening angle for tree method.
        tree : Tree, optional
            Tree structure for Barnes-Hut method.
        method : str
            Method to compute gravitational forces:
            - 'tree': Barnes-Hut tree algorithm
            - 'directSummation': O(N²) direct summation (pure Python/NumPy)
            - 'numba': Numba-accelerated direct summation
        use_numba : bool
            Whether to use Numba acceleration (for directSummation).

        Returns
        -------
        gravity_forces : array_like
            Gravitational forces on particles.
        '''
        if method == 'tree':
            return tree.force(threshold)
        elif method == 'numba' or (method == 'directSummation' and use_numba):
            # Use Numba-accelerated direct summation
            if self._numba_gravity is None:
                self._numba_gravity = NumbaGravity(use_numba=True)
            return self._numba_gravity.direct_force(q, masses, softening)
        elif method == 'directSummation':
            return self.directForceSummation(q, masses)

    def _hydroForces(self, q, masses, h, rho0, cs, gamma=7.0):
        '''
        Compute hydrodynamic forces on particles.

        Parameters
        ----------
        q : array_like
            Positions of particles.
        masses : array_like
            Particle masses.
        h : float
            Smoothing length.
        rho0 : float
            Reference density.
        cs : float
            Sound speed.
        gamma : float
            Adiabatic index (7 for water, 5/3 for ideal gas, 1 for isothermal).

        Returns
        -------
        hydro_forces : array_like
            Hydrodynamic forces on particles.
        '''
        # Use Numba-accelerated version if available
        if self._numba_hydro is not None:
            return self._numba_hydro.pressure_acceleration(q, masses, h, rho0, cs, gamma)
        else:
            return pressureAcc(q, masses, h, rho0, cs, gamma)
    
    def directForceSummation(self, q, masses):
        '''
        Compute gravitational forces using direct summation.

        Returns
        -------
        gravity_forces : array_like
            Gravitational forces on particles.
        '''
        N = q.shape[0]
        forces = np.zeros_like(q)
        for i in range(N):
            for j in range(i + 1, N):
                r_vec = q[j] - q[i]
                r_mag = np.linalg.norm(r_vec) + 1e-10  # Softening to avoid singularity
                force = G * masses[j] * masses[i] * r_vec / r_mag**3
                forces[i] += force
                forces[j] -= force
        return forces
    
    def pynbodySnap(self, timeIndex):
        '''
        Return snapshot at a timestep as a pynbody SimSnap.
        '''

        ap_pos_unit = self.positions.unit
        pyn_pos_unit = astropy_to_pynbody(ap_pos_unit)
        ap_vel_unit = self.velocities.unit
        pyn_vel_unit = astropy_to_pynbody(ap_vel_unit)
        ap_mass_unit = self.masses.unit
        pyn_mass_unit = astropy_to_pynbody(ap_mass_unit)
                                
        sim = pyn.new(self.nParticles)

        sim['pos'] = pyn.array.SimArray(self.positions[timeIndex].to(ap_pos_unit).value,\
            pyn_pos_unit)
        sim['vel'] = pyn.array.SimArray(self.velocities[timeIndex].to(ap_vel_unit).value, \
            pyn_vel_unit)
        sim['mass'] = pyn.array.SimArray(self.masses.to(ap_mass_unit).value, \
            pyn_mass_unit)
        # sim['eps'] = pyn.array.SimArray(self.params['eps'].to(ap_pos_unit).value, pyn_pos_unit)
        sim.physical_units(distance=pyn_pos_unit, velocity=pyn_vel_unit, mass=pyn_mass_unit)
        return sim
    
    def x(self, t_index):
        return self.positions[t_index, :, 0]
    
    def y(self, t_index):
        return self.positions[t_index, :, 1]
    
    def z(self, t_index):
        return self.positions[t_index, :, 2]
    
    def vx(self, t_index):
        return self.velocities[t_index, :, 0]
    
    def vy(self, t_index):
        return self.velocities[t_index, :, 1]
    
    def vz(self, t_index):
        return self.velocities[t_index, :, 2]