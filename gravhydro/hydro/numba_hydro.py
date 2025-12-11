"""
Numba-optimized SPH hydrodynamics calculations for gravHydro.

This module provides JIT-compiled versions of SPH kernel, density, and
pressure acceleration calculations for improved performance.
Falls back to pure NumPy if Numba is not available.
"""

import numpy as np

# Try to import numba, set flag if not available
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# =============================================================================
# Kernel Functions (Numba-compiled)
# =============================================================================

@njit(fastmath=True)
def cubic_spline_kernel(r, h, ndim):
    """
    Cubic spline kernel W(r, h) - single value version.
    
    Parameters
    ----------
    r : float
        Distance between particles
    h : float
        Smoothing length
    ndim : int
        Number of dimensions (1, 2, or 3)
    
    Returns
    -------
    W : float
        Kernel value
    """
    # Normalization constants
    if ndim == 1:
        sigma = 2.0 / 3.0
    elif ndim == 2:
        sigma = 10.0 / (7.0 * np.pi)
    else:  # ndim == 3
        sigma = 1.0 / np.pi
    
    norm = sigma / (h ** ndim)
    q = r / h
    
    if q <= 1.0:
        return norm * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    elif q <= 2.0:
        tmp = 2.0 - q
        return norm * 0.25 * tmp * tmp * tmp
    else:
        return 0.0


@njit(fastmath=True)
def cubic_spline_kernel_derivative(r, h, ndim):
    """
    Derivative of cubic spline kernel dW/dr - single value version.
    
    Parameters
    ----------
    r : float
        Distance between particles
    h : float
        Smoothing length
    ndim : int
        Number of dimensions
    
    Returns
    -------
    dWdr : float
        Derivative of kernel w.r.t. r
    """
    if ndim == 1:
        sigma = 2.0 / 3.0
    elif ndim == 2:
        sigma = 10.0 / (7.0 * np.pi)
    else:
        sigma = 1.0 / np.pi
    
    norm = sigma / (h ** ndim)
    q = r / h
    
    if q <= 1.0:
        return norm / h * (-3.0 * q + 2.25 * q * q)
    elif q <= 2.0:
        tmp = 2.0 - q
        return norm / h * (-0.75 * tmp * tmp)
    else:
        return 0.0


# =============================================================================
# Density Calculation (Numba-compiled)
# =============================================================================

@njit(parallel=True, fastmath=True)
def compute_densities_numba(positions, masses, h, ndim):
    """
    Compute SPH densities for all particles using Numba.
    
    rho_i = sum_j m_j * W(|r_i - r_j|, h)
    
    Parameters
    ----------
    positions : ndarray, shape (N, ndim)
        Particle positions
    masses : ndarray, shape (N,)
        Particle masses
    h : float
        Smoothing length
    ndim : int
        Number of dimensions
    
    Returns
    -------
    densities : ndarray, shape (N,)
        Computed densities
    """
    N = positions.shape[0]
    densities = np.zeros(N, dtype=np.float64)
    
    # Kernel cutoff radius (cubic spline has compact support at 2h)
    h_cut = 2.0 * h
    h_cut_sq = h_cut * h_cut
    
    for i in prange(N):
        rho_i = 0.0
        
        for j in range(N):
            # Compute distance squared
            r_sq = 0.0
            for d in range(ndim):
                diff = positions[i, d] - positions[j, d]
                r_sq += diff * diff
            
            # Only compute kernel if within cutoff
            if r_sq < h_cut_sq:
                r = np.sqrt(r_sq)
                rho_i += masses[j] * cubic_spline_kernel(r, h, ndim)
        
        densities[i] = rho_i
    
    return densities


@njit(fastmath=True)
def compute_densities_numba_serial(positions, masses, h, ndim):
    """
    Serial version of density computation (for small N or debugging).
    """
    N = positions.shape[0]
    densities = np.zeros(N, dtype=np.float64)
    
    h_cut = 2.0 * h
    h_cut_sq = h_cut * h_cut
    
    for i in range(N):
        rho_i = 0.0
        
        for j in range(N):
            r_sq = 0.0
            for d in range(ndim):
                diff = positions[i, d] - positions[j, d]
                r_sq += diff * diff
            
            if r_sq < h_cut_sq:
                r = np.sqrt(r_sq)
                rho_i += masses[j] * cubic_spline_kernel(r, h, ndim)
        
        densities[i] = rho_i
    
    return densities


# =============================================================================
# Pressure Acceleration (Numba-compiled)
# =============================================================================

@njit(parallel=True, fastmath=True)
def compute_pressure_acceleration_numba(positions, masses, densities, pressures, h, ndim):
    """
    Compute pressure acceleration for all particles using Numba.
    
    a_i = -sum_j m_j * (P_i/rho_i^2 + P_j/rho_j^2) * grad_W_ij
    
    Parameters
    ----------
    positions : ndarray, shape (N, ndim)
        Particle positions
    masses : ndarray, shape (N,)
        Particle masses  
    densities : ndarray, shape (N,)
        Particle densities
    pressures : ndarray, shape (N,)
        Particle pressures
    h : float
        Smoothing length
    ndim : int
        Number of dimensions
    
    Returns
    -------
    accelerations : ndarray, shape (N, ndim)
        Pressure accelerations
    """
    N = positions.shape[0]
    accelerations = np.zeros((N, ndim), dtype=np.float64)
    
    h_cut = 2.0 * h
    h_cut_sq = h_cut * h_cut
    
    for i in prange(N):
        P_i = pressures[i]
        rho_i = densities[i]
        rho_i_sq = rho_i * rho_i
        
        # Accumulate acceleration components
        acc = np.zeros(ndim, dtype=np.float64)
        
        for j in range(N):
            if i == j:
                continue
            
            # Compute separation vector and distance
            r_vec = np.zeros(ndim, dtype=np.float64)
            r_sq = 0.0
            for d in range(ndim):
                r_vec[d] = positions[i, d] - positions[j, d]
                r_sq += r_vec[d] * r_vec[d]
            
            # Only compute if within kernel cutoff
            if r_sq < h_cut_sq and r_sq > 0:
                r = np.sqrt(r_sq)
                
                # Kernel gradient: grad_W = dW/dr * r_hat
                dW_dr = cubic_spline_kernel_derivative(r, h, ndim)
                
                # Pressure term: (P_i/rho_i^2 + P_j/rho_j^2)
                rho_j_sq = densities[j] * densities[j]
                pressure_term = P_i / rho_i_sq + pressures[j] / rho_j_sq
                
                # Prefactor for gradient
                prefactor = -masses[j] * pressure_term * dW_dr / r
                
                for d in range(ndim):
                    acc[d] += prefactor * r_vec[d]
        
        for d in range(ndim):
            accelerations[i, d] = acc[d]
    
    return accelerations


@njit(fastmath=True)
def compute_pressure_acceleration_numba_serial(positions, masses, densities, pressures, h, ndim):
    """
    Serial version of pressure acceleration (for small N or debugging).
    """
    N = positions.shape[0]
    accelerations = np.zeros((N, ndim), dtype=np.float64)
    
    h_cut = 2.0 * h
    h_cut_sq = h_cut * h_cut
    
    for i in range(N):
        P_i = pressures[i]
        rho_i = densities[i]
        rho_i_sq = rho_i * rho_i
        
        for j in range(N):
            if i == j:
                continue
            
            r_vec = np.zeros(ndim, dtype=np.float64)
            r_sq = 0.0
            for d in range(ndim):
                r_vec[d] = positions[i, d] - positions[j, d]
                r_sq += r_vec[d] * r_vec[d]
            
            if r_sq < h_cut_sq and r_sq > 0:
                r = np.sqrt(r_sq)
                dW_dr = cubic_spline_kernel_derivative(r, h, ndim)
                
                rho_j_sq = densities[j] * densities[j]
                pressure_term = P_i / rho_i_sq + pressures[j] / rho_j_sq
                
                prefactor = -masses[j] * pressure_term * dW_dr / r
                
                for d in range(ndim):
                    accelerations[i, d] += prefactor * r_vec[d]
    
    return accelerations


# =============================================================================
# Combined Pressure Acceleration Function
# =============================================================================

@njit(fastmath=True)
def compute_pressure_from_density(densities, rho0, cs, gamma):
    """
    Compute pressure using equation of state.
    
    P = cs^2 * rho0 * ((rho/rho0)^gamma - 1) / gamma  (Tait equation)
    or P = cs^2 * (rho - rho0) for nearly incompressible
    
    For isothermal (gamma ~ 1): P = cs^2 * rho
    """
    N = len(densities)
    pressures = np.zeros(N, dtype=np.float64)
    
    if abs(gamma - 1.0) < 0.01:
        # Isothermal case
        for i in range(N):
            pressures[i] = cs * cs * densities[i]
    else:
        # General case (Tait equation of state)
        cs_sq = cs * cs
        for i in range(N):
            ratio = densities[i] / rho0
            pressures[i] = cs_sq * rho0 * (ratio ** gamma - 1.0) / gamma
    
    return pressures


def pressure_acceleration_numba(positions, masses, h, rho0, cs, gamma=7.0, use_parallel=True):
    """
    Main entry point for Numba-accelerated pressure acceleration.
    
    Parameters
    ----------
    positions : ndarray
        Particle positions, shape (N, ndim)
    masses : ndarray
        Particle masses, shape (N,)
    h : float
        Smoothing length
    rho0 : float
        Reference density
    cs : float
        Sound speed
    gamma : float
        Adiabatic index
    use_parallel : bool
        Whether to use parallel computation
    
    Returns
    -------
    accelerations : ndarray
        Pressure accelerations, shape (N, ndim)
    """
    ndim = positions.shape[1]
    
    # Ensure arrays are contiguous and float64
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    masses = np.ascontiguousarray(masses, dtype=np.float64)
    
    # Compute densities
    if use_parallel:
        densities = compute_densities_numba(positions, masses, h, ndim)
    else:
        densities = compute_densities_numba_serial(positions, masses, h, ndim)
    
    # Compute pressures from equation of state
    pressures = compute_pressure_from_density(densities, rho0, cs, gamma)
    
    # Compute accelerations
    if use_parallel:
        accelerations = compute_pressure_acceleration_numba(
            positions, masses, densities, pressures, h, ndim
        )
    else:
        accelerations = compute_pressure_acceleration_numba_serial(
            positions, masses, densities, pressures, h, ndim
        )
    
    return accelerations


# =============================================================================
# Neighbor List for O(N) Scaling
# =============================================================================

@njit(fastmath=True)
def build_cell_list(positions, h, ndim):
    """
    Build a cell list for efficient neighbor finding.
    
    Cell size = 2h (kernel cutoff), so only need to search 
    neighboring cells for interactions.
    
    Returns cell indices for each particle.
    """
    N = positions.shape[0]
    cell_size = 2.0 * h
    
    # Find bounds
    mins = np.empty(ndim, dtype=np.float64)
    maxs = np.empty(ndim, dtype=np.float64)
    for d in range(ndim):
        mins[d] = positions[0, d]
        maxs[d] = positions[0, d]
        for i in range(1, N):
            if positions[i, d] < mins[d]:
                mins[d] = positions[i, d]
            if positions[i, d] > maxs[d]:
                maxs[d] = positions[i, d]
    
    # Compute number of cells in each dimension
    n_cells = np.empty(ndim, dtype=np.int64)
    for d in range(ndim):
        n_cells[d] = max(1, int(np.ceil((maxs[d] - mins[d]) / cell_size)))
    
    # Compute cell index for each particle
    cell_indices = np.empty(N, dtype=np.int64)
    for i in range(N):
        idx = 0
        multiplier = 1
        for d in range(ndim):
            cell_d = int((positions[i, d] - mins[d]) / cell_size)
            cell_d = min(cell_d, n_cells[d] - 1)  # Handle edge case
            idx += cell_d * multiplier
            multiplier *= n_cells[d]
        cell_indices[i] = idx
    
    return cell_indices, mins, n_cells, cell_size


def is_numba_available():
    """Check if Numba is available."""
    return NUMBA_AVAILABLE


# =============================================================================
# Wrapper class similar to NumbaGravity
# =============================================================================

class NumbaHydro:
    """
    Class to manage Numba-accelerated hydrodynamics calculations.
    
    Similar interface to NumbaGravity for consistency.
    """
    
    def __init__(self, use_numba=True):
        """
        Initialize the NumbaHydro calculator.
        
        Parameters
        ----------
        use_numba : bool
            Whether to use Numba. Will fall back to NumPy if False
            or if Numba is not available.
        """
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self._first_call = True
    
    def pressure_acceleration(self, positions, masses, h, rho0, cs, gamma=7.0):
        """
        Compute pressure acceleration.
        
        On first call, triggers JIT compilation.
        """
        if self._first_call and self.use_numba:
            # Warm up JIT compilation with a small test
            test_pos = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
            test_mass = np.array([1.0, 1.0], dtype=np.float64)
            _ = pressure_acceleration_numba(test_pos, test_mass, 1.0, 1.0, 1.0, gamma)
            self._first_call = False
        
        if self.use_numba:
            return pressure_acceleration_numba(positions, masses, h, rho0, cs, gamma)
        else:
            # Fall back to original implementation
            from .acceleration import pressureAcc
            return pressureAcc(positions, masses, h, rho0, cs, gamma)
