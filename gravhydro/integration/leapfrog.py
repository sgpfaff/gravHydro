
import numpy as np

def leapfrog(q, p, masses, dt, force, nsteps):
    for _ in range(nsteps):
        q, p = leapfrogStep(q, p, masses, dt, force)
    return q, p

def leapfrogStep(q, p , masses, dt, force):
    q_half = leapfrogLeap_q(q, p, masses, dt/2)
    p_full = leapfrogLeap_p(q_half, p, dt, force)
    q_full = leapfrogLeap_q(q_half, p_full, masses, dt/2)
    return q_full, p_full

def leapfrogLeap_q(q, p, masses, dt):
    #print(p.shape, masses)
    return q + (p/masses[:, np.newaxis]) * dt

def leapfrogLeap_p(q, p, dt, force):
    return p + force * dt