from .generalProperty import propertyGradient_i, propertyGradients
from .density import densities, density_i
from .pressure import pressure
from .kernel import gradW
import numpy as np


def pressure_acceleration_i(i, positions, masses, pressures, dens, h):
    """
    Calculate pressure acceleration on particle i using standard SPH formulation:
    
    a_i = -sum_j m_j * (P_i/rho_i^2 + P_j/rho_j^2) * grad_W_ij
    """
    r_i = positions[i]
    r_ij = r_i - positions  # vector from j to i
    r_mag = np.linalg.norm(r_ij, axis=1)
    
    # Mask out self-interaction
    mask = r_mag > 0
    
    if not np.any(mask):
        return np.zeros(positions.shape[1])
    
    r_ij_masked = r_ij[mask]
    masses_masked = masses[mask]
    P_j = pressures[mask]
    rho_j = dens[mask]
    
    # Kernel gradient
    grad_W = gradW(r_ij_masked, h)
    
    # SPH pressure term: (P_i/rho_i^2 + P_j/rho_j^2)
    P_i = pressures[i]
    rho_i = dens[i]
    pressure_term = P_i / rho_i**2 + P_j / rho_j**2
    
    # Acceleration: -sum_j m_j * pressure_term * grad_W
    acc = -np.sum(masses_masked[:, np.newaxis] * pressure_term[:, np.newaxis] * grad_W, axis=0)
    
    return acc


def pressureAcc(positions, masses, h, rho0, cs, gamma=5/3):
    '''
    Calculate the accelerations at the 
    positions provided caused by pressure force.
    
    Uses the standard SPH momentum equation:
    a_i = -sum_j m_j * (P_i/rho_i^2 + P_j/rho_j^2) * grad_W_ij
    
    Parameters
    ----------
    gamma : float
        Adiabatic index. Use 7 for water/incompressible, 
        5/3 for monatomic ideal gas, 1.0001 for nearly isothermal.
    '''
    dens = densities(positions, masses, h)
    pressures = pressure(dens, rho0, cs, gamma=gamma)
    
    n_particles = len(positions)
    accelerations = np.zeros_like(positions)
    
    for i in range(n_particles):
        accelerations[i] = pressure_acceleration_i(i, positions, masses, pressures, dens, h)
    
    return accelerations

def pressureAcceleration_i(pos_i, positions, masses, densities, pressures, h, rho0, cs, gamma):
    '''Acceleration due to presure on a specific particle.'''
    r_ij = pos_i - positions  # vector from j to i
    r_mag = np.linalg.norm(r_ij, axis=1)
    
    # Kernel gradient
    grad_W = gradW(positions, h)
    
    # SPH pressure term: (P_i/rho_i^2 + P_j/rho_j^2)
    rho_i = density_i(pos_i, positions, masses, h)
    P_i = pressure(densities, rho0, cs, gamma)
    pressure_term = P_i / rho_i**2 + pressures / densities**2
    
    # Acceleration: -sum_j m_j * pressure_term * grad_W
    acc = -np.sum(masses[:, np.newaxis] * pressure_term[:, np.newaxis] * grad_W, axis=0)
    
    return acc

# Keep old functions for backwards compatibility but mark as deprecated
def acc_inside_term_ij(i, j, pressures, densities):
    """DEPRECATED: Use pressure_acceleration_i instead."""
    return (pressures[i] / densities[i]**2) + (pressures[j] / densities[j]**2)

def acc_inside_term_i(i, pressures, densities):
    """DEPRECATED: This function was incorrect."""
    total_acc = 0
    for j in range(len(pressures)):
        if i == j:
            continue
        else:
            total_acc += acc_inside_term_ij(i, j, pressures, densities)
    return total_acc

def acc_inside_terms(pressures, densities):
    """DEPRECATED: This function was incorrect."""
    inside_terms = np.zeros_like(pressures)
    for i in range(len(pressures)):
        inside_terms[i] = acc_inside_term_i(i, pressures, densities)
    return inside_terms