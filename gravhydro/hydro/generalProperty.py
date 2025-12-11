from .kernel import gradW
import numpy as np

def propertyGradient_i(position_i, positions, masses, densities, properties, h):
    '''
    Calculate the gradient of a property at position r_i.
    '''
    r_ij = position_i - positions
    mask = np.linalg.norm(r_ij, axis=1) > 0
    r_ij = r_ij[mask]
    masses = masses[mask]
    densities = densities[mask]
    properties = properties[mask]
    gradW_ij = gradW(r_ij, h)  # gradient of kernel function
    prop_grad = np.sum(masses[:, np.newaxis] * properties[:, np.newaxis] * gradW_ij / densities[:,np.newaxis], axis=0)
    return prop_grad

def propertyGradients(positions, masses, densities, values, h):
    gradients = np.zeros_like(positions)
    for i, r_i in enumerate(positions):
        gradients[i] = propertyGradient_i(r_i, positions, masses, densities, values, h)
    return gradients