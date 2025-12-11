
import numpy as np

def total_acceleration():
    return a_pressure() + a_viscosity() + a_external()
def a_pressure():
    pass
def a_viscosity():
    pass
def a_external():
    pass


# def pressure(density, ref_density, sound_speed):
#     '''
#     Parameters
#     ----------
#     density : array_like
#         Density for particle of interest.
#     ref_density : float
#         fluid reference density.
#     sound_speed : float
#         fluid sound speed.
#         '''
#     return sound_speed**2 * (density - ref_density)
# def density(r_i, rs, masses kernel):
#     return np.sum(masses * kernel(r_i - rs))
