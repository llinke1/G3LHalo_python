from scipy.special import erf
import numpy as np

def HOD_Zheng(alpha, mth, sigma, mprime, beta):
    """Zheng 2007-like HOD with varying central galaxy fraction

    Args:
        alpha (float): central galaxy fraction
        mth (float): Threshold mass for central galaxies [Msun]
        sigma (float): Width of central galaxy HOD
        mprime (float): Mass scale for satellite galaxies [Msun]
        beta (float): Power-law index for satellite distribution

    Returns:
        function: Central galaxy number
        function: Satellite galaxy number
    """
    Ncen = lambda m: 0.5*alpha*(1+erf(np.log(m/mth)/np.sqrt(2)/sigma))

    Nsat= lambda m: Ncen(m)/alpha*np.power(m/mprime, beta)
    
    return Ncen, Nsat


def HOD_Zheng05(mmin, sigma, m0, m1, alpha):
    """Zheng 2005 HOD

    Args:
        mmin (float): Minimal mass for central galaxies [Msun]
        sigma (float): Width of central galaxy HOD
        m0 (float): Minimal mass for satellite galaxies [Msun]
        m1 (float): Mass scale for satellite galaxies [Msun]
        alpha (float): Power-law index for satellite distribution

    Returns:
        function: Central galaxy number
        function: Satellite galaxy number
    """
    Ncen = lambda m: 0.5*(1+erf(np.log10(m/mmin)/sigma))
    Nsat = lambda m: (m>m0)*np.power((m-m0)/m1, alpha)


    return Ncen, Nsat