
def leapfrog(q, p, dt, force, nsteps):
    for _ in range(nsteps):
        q, p = leapfrogStep(q, p, dt, force)
    return q, p

def leapfrogStep(q, p , dt, force):
    q_half = leapfrogLeap_q(q, p, dt/2)
    p_full = leapfrogLeap_p(q_half, p, dt, force)
    q_full = leapfrogLeap_q(q_half, p_full, dt/2)
    return q_full, p_full

def leapfrogLeap_q(q, p, dt):
    return q + p * dt

def leapfrogLeap_p(q, p, dt, force):
    return p + force * dt