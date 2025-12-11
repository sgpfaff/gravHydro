import numpy as np

def W(r, h, ndim=3):
    '''
    Cubic spline kernel function with proper normalization.

    Parameters
    ----------
    r : array of shape [N]
        separation(s) between particles.
    h : float
        smoothing length.
    ndim : int
        number of spatial dimensions (1, 2, or 3). Default is 3.

    Returns
    -------
    Ws : array of shape [N]
        Value(s) of the kernel function for the
        given particle separations and smoothing
        length.
    '''
    # Normalization constants for the cubic spline kernel
    # These ensure the kernel integrates to 1 in ndim dimensions
    if ndim == 1:
        sigma = 2.0 / 3.0
    elif ndim == 2:
        sigma = 10.0 / (7.0 * np.pi)
    else:  # ndim == 3
        sigma = 1.0 / np.pi
    
    norm = sigma / h**ndim
    
    q = r / h  # Standard definition: q = r/h, kernel support is q <= 2
    first_mask = (q >= 0) & (q <= 1)
    second_mask = (q > 1) & (q <= 2)
    third_mask = ~(first_mask | second_mask)
    
    Ws = np.zeros_like(r)
    Ws[first_mask] = norm * (1 - 1.5 * q[first_mask]**2 + 0.75 * q[first_mask]**3)
    Ws[second_mask] = norm * 0.25 * (2 - q[second_mask])**3
    Ws[third_mask] = 0
    return Ws

def dWdr(r, h, ndim=3):
    '''
    Derivative of the kernel function W with respect to r.
    
    Parameters
    ----------
    r : array
        separation between particles.
    h : float
        smoothing length
    ndim : int
        number of spatial dimensions (1, 2, or 3). Default is 3.

    Returns
    -------
    dWdr : array
        Value(s) of the derivative of
        the kernel function for the
        given particle separations 
        and smoothing length.
    
    '''
    # Normalization constants
    if ndim == 1:
        sigma = 2.0 / 3.0
    elif ndim == 2:
        sigma = 10.0 / (7.0 * np.pi)
    else:  # ndim == 3
        sigma = 1.0 / np.pi
    
    norm = sigma / h**ndim
    
    q = r / h
    first_mask = (q >= 0) & (q <= 1)
    second_mask = (q > 1) & (q <= 2)
    third_mask = ~(first_mask | second_mask)
    
    dWdr_arr = np.zeros_like(r)
    # Derivative of kernel w.r.t. r: dW/dr = (dW/dq) * (dq/dr) = (dW/dq) / h
    dWdr_arr[first_mask] = norm / h * (-3 * q[first_mask] + 2.25 * q[first_mask]**2)
    dWdr_arr[second_mask] = norm / h * (-0.75 * (2 - q[second_mask])**2)
    dWdr_arr[third_mask] = 0
    return dWdr_arr

def gradW(r_vec, h, ndim=None):
    '''
    Calculates the gradient of the kernel function
    for a particular separation vector.

    Parameters
    ----------
    r_vec : array of shape [N, d]
        vector separation(s) (of dimension d) 
        between N particles.
    h : float
        smoothing length.
    ndim : int, optional
        number of spatial dimensions. If None, inferred from r_vec.

    Returns
    -------
    gradWs : array of shape [N, d]
        gradient of the kernel function for the
        given vector separations and smoothing
        length.
    '''
    if ndim is None:
        ndim = r_vec.shape[1]
    
    r = np.linalg.norm(r_vec, axis=1)
    dW_dr = dWdr(r, h, ndim=ndim)
    
    # Handle division by zero for r=0
    with np.errstate(divide='ignore', invalid='ignore'):
        r_hat = r_vec / r[:, np.newaxis]
        r_hat = np.nan_to_num(r_hat, nan=0.0)
    
    return dW_dr[:, np.newaxis] * r_hat