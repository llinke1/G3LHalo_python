from scipy.special import erf
import numpy as np

def HOD_Zheng(alpha, mth, sigma, mprime, beta):
    Ncen = lambda m: 0.5*alpha*(1+erf(np.log(m/mth)/np.sqrt(2)/sigma))

    Nsat= lambda m: Ncen(m)/alpha*np.power(m/mprime, beta)
    
    return Ncen, Nsat


def HOD_Zheng05(mmin, sigma, m0, m1, alpha):
    Ncen = lambda m: (m>m0)*np.power((m-m0)/m1, alpha)
    Nsat = lambda m: 0.5*(1+erf(np.log10(m/mmin)/sigma))

    return Ncen, Nsat