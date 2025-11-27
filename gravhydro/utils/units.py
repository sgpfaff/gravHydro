import astropy.units as u
import astropy.constants as c
import numpy as np
import pynbody.units as pynu
from fractions import Fraction

G_IN = 1
G_SI = c.G
L_UNIT = 100 * u.pc
M_UNIT = 1e4 * u.Msun
t_UNIT = np.sqrt(G_IN * (L_UNIT**3) / (M_UNIT * G_SI)).to(u.Myr)


def convert_to_internal(input):
    '''convert input to internal units'''
    if isinstance(input, u.Quantity):
        # Convert length
        if input.unit.physical_type == 'length':
            return (input / L_UNIT).decompose().value
        # Convert mass
        elif input.unit.physical_type == 'mass':
            return (input / M_UNIT).decompose().value
        # Convert time
        elif input.unit.physical_type == 'time':
            return (input / t_UNIT).decompose().value
        # Convert velocity
        elif input.unit.physical_type == 'speed':
            return (input / (L_UNIT / t_UNIT)).decompose().value
        # Convert density
        elif input.unit.physical_type == 'mass density':
            return (input / (M_UNIT / L_UNIT**3)).decompose().value
        else:
            raise ValueError(f"Unsupported unit type: {input.unit.physical_type}")
    else:
        # If no units, assume already in internal units
        return input

def convert_to_physical(input, type, returnAstropy):
    if type == 'length':
        converted = (input * L_UNIT).to(u.kpc)
    elif type == 'speed':
        converted = (input * (L_UNIT / t_UNIT)).to(u.km / u.s)
    else:
        raise ValueError(f"Unsupported unit type: {type}")
    
    if returnAstropy:
        return converted
    else:
        return converted.value


### The following code was lifted from gravhopper's unitconverter.py ###

unitmapper_pyn2ap = {
    'm':u.m,
    's':u.s,
    'kg':u.kg,
    'K':u.K,
    'rad':u.radian,
    'yr':u.yr,
    'kyr':u.kyr,
    'Myr':u.Myr,
    'Gyr':u.Gyr,
    'Hz':u.Hz,
    'kHz':u.kHz,
    'MHz':u.MHz,
    'GHz':u.GHz,
    'THz':u.THz,
    'angst':u.AA,
    'cm':u.cm,
    'mm':u.mm,
    'nm':u.nm,
    'km':u.km,
    'au':u.au,
    'pc':u.pc,
    'kpc':u.kpc,
    'Mpc':u.Mpc,
    'Gpc':u.Gpc,
    'sr':u.sr,
    'deg':u.deg,
    'arcmin':u.arcmin,
    'arcsec':u.arcsec,
    'Msol':u.Msun,
    'g':u.g,
    'm_p':c.m_p,
    'm_e':c.m_e,
    'N':u.N,
    'dyn':u.dyn,
    'J':u.J,
    'erg':u.erg,
    'eV':u.eV,
    'keV':u.keV,
    'MeV':u.MeV,
    'W':u.W,
    'Jy':u.Jy,
    'Pa':u.Pa,
    'k':c.k_B,
    'c':c.c,
    'G':c.G,
    'hP':c.h
}
# Invert mapping from astropy units to pynbody base equivalents
unitmapper_ap2pyn = {}
for k, v in unitmapper_pyn2ap.items():
    # If it's a unit, use the Astropy string repr, otherwise use the Pynbody symbol too.
    if isinstance(v, u.UnitBase):
        unitmapper_ap2pyn[str(v)] = k
    else:
        unitmapper_ap2pyn[k] = k


def astropy_to_pynbody(apunit):
    """Return a pynbody unit equivalent to the astropy unit input.
    
    Parameters
    ----------
    apunit : astropy.unit.Unit
        Astropy unit to convert
        
    Returns
    -------
    pynunit : pynbody.units.Unit
        Equivalent Pynbody unit. If the input unit is a composite, the function will attempt
        to compose it in the same way, but sometimes that is not possible; however, it
        will always be equivalent numerically.
        
    Raises
    ------
    pynbody.UnitsException
        If it cannot successfully convert the unit
    """
#     See also
#     --------
#     `pynbody_to_astropy` : Converts astropy units to pynbody units.
#     :meth:`~gravhopper.Simulation.pyn_snap` : Converts a Simulation snapshot to a Pynbody SimSnap
#     """
    
    # Emergency base units for each dimension.
    defaults = (u.kpc, u.s, u.Msun, u.K, u.radian, u.sr)
        
    # Does it have a direct pynbody equivalent?
    apunit_str = str(apunit)
    if apunit_str in unitmapper_ap2pyn:
        return pynu.Unit(unitmapper_ap2pyn[apunit_str])
    
    # If it's a quantity, split into scale and unit
    if isinstance(apunit, u.Quantity):
        scale = apunit.value
        decomp_unit = apunit.unit
    elif isinstance(apunit, (u.Unit, u.CompositeUnit)):
        scale = apunit.scale
        decomp_unit = apunit
    else:
        raise pynu.UnitsException(f"Error decomposing Astropy unit {apunit}")
    
    # Construct a CompositeUnit
    pynbody_equivalent = pynu.CompositeUnit(scale, [], [])
    
    # Go through each decomposed piece
    for base, power in zip(decomp_unit.bases, decomp_unit.powers):
        # See if each base has a pynbody equivalent
        base_str = str(base)
        if base_str in unitmapper_ap2pyn:
            compose_pynbody_unit(pynbody_equivalent, 1, [base], [power])
        elif isinstance(base, u.PrefixUnit):
            # If it's a prefix unit, decompose it and try again
            prefix_decomp = base.decompose()
            compose_pynbody_unit(pynbody_equivalent, prefix_decomp.scale, prefix_decomp.bases, prefix_decomp.powers)        
        else:
            # Decompose into emergency bases
            default_decomp = base.decompose(defaults)
            compose_pynbody_unit(pynbody_equivalent, default_decomp.scale, default_decomp.bases, default_decomp.powers)
                                
    return pynbody_equivalent

def compose_pynbody_unit(unit_so_far, scale, bases, powers):
    """Add bases and powers and scale to existing pynbody composite unit."""
    
    unit_so_far._scale *= scale
    for base, power in zip(bases, powers):
        # See if each base has a pynbody equivalent
        base_str = str(base)
        if base_str in unitmapper_ap2pyn:
            pyn_base = pynu.Unit(unitmapper_ap2pyn[base_str])
            unit_so_far._bases.append(pyn_base)
            if isinstance(power, float):
                fracpower = Fraction(power)
            else:
                fracpower = power
            unit_so_far._powers.append(fracpower)
        else:
            raise pynu.UnitsException(f"Error decomposing Astropy unit {base_str}")
        
