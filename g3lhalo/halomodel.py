import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt


c_over_H0=3000 #Mpc/h


class halomodel:

    def __init__ (self, kmin, kmax, zmin, zmax, mmin, mmax, nbins=128):
        self.kmin=kmin
        self.kmax=kmax
        self.zmin=zmin
        self.zmax=zmax
        self.mmin=mmin
        self.mmax=mmax
        self.nbins=nbins


    def set_cosmo(self, Omega_c, Omega_b, h, sigma8, n_s ):
        ks=np.geomspace(self.kmin, self.kmax, self.nbins)
        zs=np.linspace(self.zmin, self.zmax, self.nbins)

        a=1/(1+zs)

        self.cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, 
                                   h=h, sigma8=sigma8, n_s=n_s)
        
        self.pk_lin=ccl.linear_matter_power(self.cosmo, ks, a)
        self.w=ccl.comoving_angular_distance(self.cosmo, a)
        self.dwdz=1/np.sqrt(ccl.h_over_h0(self.cosmo, a))*c_over_H0

    

    def set_nz_lenses(self, zs, n):
        zs_new=np.linspace(self.zmin, self.zmax, self.nbins)
        self.n_l=np.interp(zs_new, zs, n, left=0, right=0)
        plt.plot(zs, n)
        plt.plot(zs_new, self.n_l)

    
    def set_nz_sources(self, zs, n):
        zs_new=np.linspace(self.zmin, self.zmax, self.nbins)
        self.n_s=np.interp(zs_new, zs, n, left=0, right=0)
        self.set_lensing_efficiency()


    def set_lensing_efficiency(self):
        self.gs=np.zeros_like(self.w)
        dz=(self.zmax-self.zmin)/self.nbins

        for i, w in enumerate(self.w):
            for j, wprime in enumerate(self.w):
                if wprime>=w:
                    self.gs[i]+=self.n_s[j]*(wprime-w)/wprime*dz



    def set_hmf(self, hmfunc):
        ms=np.geomspace(self.mmin, self.mmax, self.nbins)
        zs=np.linspace(self.zmin, self.zmax, self.nbins)
        a=1/(1+zs)

        self.dndm=[]
        for a_ in a:
            self.dndm.append(hmfunc(self.cosmo, ms, a_)/(ms*np.log(10)))

        self.dndm=np.array(self.dndm)

    def set_halobias(self, hbfunc):
        ms=np.geomspace(self.mmin, self.mmax, self.nbins)
        zs=np.linspace(self.zmin, self.zmax, self.nbins)
        a=1/(1+zs)

        self.bh=[]
        for a_ in a:
            self.bh.append(hbfunc(self.cosmo, ms, a_))

        self.bh=np.array(self.bh)



    def set_concentration_mass_relation(self, cmfunc):
        ms=np.geomspace(self.mmin, self.mmax, self.nbins)
        zs=np.linspace(self.zmin, self.zmax, self.nbins)
        a=1/(1+zs)

        self.cm=[]
        for a_ in a:
            self.cm.append(cmfunc(self.cosmo, ms, a_))

        self.cm=np.array(self.cm)



    def set_hod1(self, hodfunc_cen, hodfunc_sat):
        ms=np.geomspace(self.mmin, self.mmax, self.nbins)

        self.hod_cen1=hodfunc_cen(ms)
        self.hod_sat1=hodfunc_sat(ms)

    def set_hod2(self, hodfunc_cen, hodfunc_sat):
        ms=np.geomspace(self.mmin, self.mmax, self.nbins)

        self.hod_cen2=hodfunc_cen(ms)
        self.hod_sat2=hodfunc_sat(ms)

    def set_hod_corr(self, A=-1, epsilon=0):
        ms=np.geomspace(self.mmin, self.mmax, self.nbins)
        
        self.hod_satsat=self.hod_sat1*self.hod_sat2+A*np.power(ms, epsilon)*np.sqrt(self.hod_sat1*self.hod_sat2)

    def set_hods(self, hodfunc_cen1, hodfunc_sat1, hodfunc_cen2=None, hodfunc_sat2=None, A=-1, epsilon=0, flens1=1, flens2=1):
        self.set_hod1(hodfunc_cen1, hodfunc_sat1)
        if hodfunc_cen2!=None:
            self.set_hod2(hodfunc_cen2, hodfunc_sat2)

        self.set_hod_corr(A, epsilon)

        self.flens1=flens1
        self.flens2=flens2

    

    def lens_lens_ps_1h

    def lens_lens_ps_2h

    def lens_lens_ps



    def source_lens_ps_1h

    def source_lens_ps_2h

    def source_lens_ps


    def source_lens_lens_bs_1h

    def source_lens_lens_bs_2h

    def source_lens_lens_bs_3h

    def source_lens_lens_bs


    def source_source_lens_bs_1h

    def source_source_lens_bs_2h

    def source_source_lens_bs_3h

    def source_source_lens_bs