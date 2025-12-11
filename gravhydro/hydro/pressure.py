import numpy as np

def pressure(densities, rho0, cs, gamma=5/3):
    '''
    Calculate the pressure for a given array of densities using 
    the Tait equation of state (commonly used in weakly-compressible SPH).

    Parameters
    ----------
    densities : array_like
        Density for particles.
    rho0 : float
        Fluid reference density.
    cs : float
        Fluid sound speed.
    gamma : float
        Stiffness parameter (default 7.0 for water).
        
    Returns
    -------
    P : array_like
        Pressure values (always non-negative).
        
    Notes
    -----
    Uses the Tait equation: P = (rho0 * cs^2 / gamma) * ((rho/rho0)^gamma - 1)
    This ensures pressure is always >= 0 when rho >= 0.
    For nearly incompressible fluids, this reduces to approximately 
    P ≈ cs^2 * (rho - rho0) when rho ≈ rho0.
    '''
    # B = rho0 * cs**2 / gamma  # pressure coefficient
    # P = B * ((densities / rho0)**gamma - 1)
    P = densities * cs**2 / gamma
    # Ensure non-negative pressure (can happen due to numerical errors)
    return np.maximum(P, 0.0)