import numpy as np
import pyccl as ccl
from scipy import interpolate

class projectedSpectra:

    def __init__(self, halomod):
        self.cosmo=halomod.cosmo
        self.halomod=halomod
        self.z_bins=np.linspace(0, 2)
        self.zs=0.5*(self.z_bins[1:]+self.z_bins[:1])
        self.deltaZs=self.z_bins[1:]-self.z_bins[:1]

        self.ws=ccl.comoving_angular_distance(self.cosmo, 1/(1+self.zs))

    def set_nz_lenses(self, zs, n):
        self.n_l = interpolate.interp1d(zs, n, fill_value=0)

    def set_nz_sources(self, zs, n):
        self.n_s = interpolate.interp1d(zs, n, fill_value=0)
        self.set_lensing_efficiency()

    def set_lensing_efficiency(self):
        self.gs = np.zeros_like(self.zs)

        for i, z in enumerate(self.zs):
            for j, zprime in enumerate(self.zs):
                if zprime >= z:
                    self.gs[i] += (1+z)*self.n_s(zprime)*(self.ws[j]-self.ws[i])/self.ws[i]
        
        h=self.cosmo._params['H0']/100
        Om=self.cosmo._params['Omega_c']+self.cosmo._params['Omega_b']
        HubbleRadius= 2998/h #self.cosmo.hubble_distance(1)
        self.gs*=3/2*Om/HubbleRadius/HubbleRadius

    def C_kk(self, ell):
        ell=np.array(ell)
        result_1h = np.zeros_like(ell)
        result_2h = np.zeros_like(ell)

        for i, z in enumerate(self.zs):
            P1h, P2h, P_all=self.halomod.source_source_ps(ell/self.ws[i], z)

            result_1h+=self.gs[i]**2*P1h
            result_2h+=self.gs[i]**2*P2h

        return result_1h, result_2h, result_1h+result_2h

    def C_kg(self, ell, type=1):
        ell=np.array(ell)
        result_1h = np.zeros_like(ell)
        result_2h = np.zeros_like(ell)

        for i, z in enumerate(self.zs):
            P1h, P2h, P_all=self.halomod.source_lens_ps(ell/self.ws[i], z, type)

            result_1h+=self.gs[i]/self.ws[i]*self.n_l(z)*P1h
            result_2h+=self.gs[i]/self.ws[i]*self.n_l(z)*P2h

        return result_1h, result_2h, result_1h+result_2h
    

    def C_gg(self, ell, type1=1, type2=1):
        ell=np.array(ell)
        result_1h = np.zeros_like(ell)
        result_2h = np.zeros_like(ell)

        for i, z in enumerate(self.zs):
            P1h, P2h, P_all=self.halomod.lens_lens_ps(ell/self.ws[i], z, type1, type2)

            result_1h+=self.n_l(z)**2/self.ws[i]**2*P1h
            result_2h+=self.n_l(z)**2/self.ws[i]**2*P2h

        return result_1h, result_2h, result_1h+result_2h

    
    def C_kgg(self, ell1, ell2, ell3, type1=1, type2=1):

        result_1h=0
        result_2h=0
        result_3h=0

        for i, z in enumerate(self.zs):
            w=self.ws[i]
            P1h, P2h, P3h, _ = self.halomod.source_lens_lens_bs(ell1/w, ell2/w, ell3/w, z, type1, type2)

            result_1h+=self.gs[i]**2*self.n_l(z)/w/w*P1h
            result_2h+=self.gs[i]**2*self.n_l(z)/w/w*P2h
            result_3h+=self.gs[i]**2*self.n_l(z)/w/w*P3h
        return result_1h, result_2h, result_3h


    def C_kkg(self, ell1, ell2, ell3, type=1):

        result_1h=0
        result_2h=0
        result_3h=0

        for i, z in enumerate(self.zs):
            w=self.ws[i]
            P1h, P2h, P3h, _ = self.halomod.source_source_lens_bs(ell1/w, ell2/w, ell3/w, z, type)

            result_1h+=self.gs[i]*self.n_l(z)**2/w/w/w*P1h
            result_2h+=self.gs[i]*self.n_l(z)**2/w/w/w*P2h
            result_3h+=self.gs[i]*self.n_l(z)**2/w/w/w*P3h
        return result_1h, result_2h, result_3h
