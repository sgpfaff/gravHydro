import numpy as np
from .kernel import W

def density_i(r, rs, masses, h):
    r'''
    Calculate density of a single particle.
    
    .. math::
        \\rho_i = \\sum_j m_j W(||r_i - r_j||, h)
    ...
    '''
    return np.sum(masses * W(np.linalg.norm(r - rs, axis=1), h))

# TO FIX: vectorize this function and remove self-summing
def densities(positions, masses, h):
    dens = np.zeros(positions.shape[0])
    for i, pos in enumerate(positions):
        dens[i] = density_i(pos, positions, masses, h)
    return dens