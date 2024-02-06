import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sici
import scipy.integrate as integrate
#from quadpy import quad

c_over_H0 = 3000  # Mpc/h


class halomodel:

    def __init__(self, kmin, kmax, zmin, zmax, mmin, mmax, nbins=128):
        self.kmin = kmin
        self.kmax = kmax
        self.zmin = zmin
        self.zmax = zmax
        self.mmin = mmin
        self.mmax = mmax
        self.nbins = nbins

    def set_cosmo(self, Omega_c, Omega_b, h, sigma8, n_s):
        ks = np.geomspace(self.kmin, self.kmax, self.nbins)
        zs = np.linspace(self.zmin, self.zmax, self.nbins)

        a = 1 / (1 + zs)

        self.cosmo = ccl.Cosmology(
            Omega_c=Omega_c, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s
        )

        self.pk_lin = ccl.linear_matter_power(self.cosmo, ks, a)
        self.w = ccl.comoving_angular_distance(self.cosmo, a)
        self.dwdz = 1 / np.sqrt(ccl.h_over_h0(self.cosmo, a)) * c_over_H0

    def set_nz_lenses(self, zs, n):
        zs_new = np.linspace(self.zmin, self.zmax, self.nbins)
        self.n_l = np.interp(zs_new, zs, n, left=0, right=0)
        plt.plot(zs, n)
        plt.plot(zs_new, self.n_l)

    def set_nz_sources(self, zs, n):
        zs_new = np.linspace(self.zmin, self.zmax, self.nbins)
        self.n_s = np.interp(zs_new, zs, n, left=0, right=0)
        self.set_lensing_efficiency()

    def set_lensing_efficiency(self):
        self.gs = np.zeros_like(self.w)
        dz = (self.zmax - self.zmin) / self.nbins

        for i, w in enumerate(self.w):
            for j, wprime in enumerate(self.w):
                if wprime >= w:
                    self.gs[i] += self.n_s[j] * (wprime - w) / wprime * dz

    def set_hmf(self, hmfunc):
        #ms = np.geomspace(self.mmin, self.mmax, self.nbins)
        #zs = np.linspace(self.zmin, self.zmax, self.nbins)
        #a = 1 / (1 + zs)

        #self.dndm = []
        #for a_ in a:
        #    self.dndm.append(hmfunc(self.cosmo, ms, a_) / (ms * np.log(10)))

        #self.dndm = np.array(self.dndm)
        self.dndm=lambda m,z:hmfunc(self.cosmo, m, 1./(1.+z))/(m*np.log(10))

    def set_halobias(self, hbfunc):
        ms = np.geomspace(self.mmin, self.mmax, self.nbins)
        zs = np.linspace(self.zmin, self.zmax, self.nbins)
        a = 1 / (1 + zs)

        self.bh = []
        for a_ in a:
            self.bh.append(hbfunc(self.cosmo, ms, a_))

        self.bh = np.array(self.bh)

    def set_concentration_mass_relation(self, cmfunc):
        self.cm = cmfunc
        # ms=np.geomspace(self.mmin, self.mmax, self.nbins)
        # zs=np.linspace(self.zmin, self.zmax, self.nbins)
        # a=1/(1+zs)

        # self.cm=[]
        # for a_ in a:
        #     self.cm.append(cmfunc(self.cosmo, ms, a_))

        # self.cm=np.array(self.cm)

    def set_hod1(self, hodfunc_cen, hodfunc_sat):
        # ms=np.geomspace(self.mmin, self.mmax, self.nbins)

        # self.hod_cen1=hodfunc_cen(ms)
        # self.hod_sat1=hodfunc_sat(ms)

        self.hod_cen1 = hodfunc_cen
        self.hod_sat1 = hodfunc_sat

    def set_hod2(self, hodfunc_cen, hodfunc_sat):
        # ms=np.geomspace(self.mmin, self.mmax, self.nbins)

        # self.hod_cen2=hodfunc_cen(ms)
        # self.hod_sat2=hodfunc_sat(ms)

        self.hod_cen2 = hodfunc_cen
        self.hod_sat2 = hodfunc_sat

    def set_hod_corr(self, A=-1, epsilon=0):
        # ms = np.geomspace(self.mmin, self.mmax, self.nbins)
        # self.hod_satsat = self.hod_sat1 * self.hod_sat2 + A * np.power(
        #     ms, epsilon
        # ) * np.sqrt(self.hod_sat1 * self.hod_sat2)

        self.hod_satsat=lambda m: self.hod_sat1(m)*self.hod_sat2(m) + A*np.power(m, epsilon) * np.sqrt(self.hod_sat1(m)*self.hod_sat2(m))


    def set_hods(
        self,
        hodfunc_cen1,
        hodfunc_sat1,
        hodfunc_cen2=None,
        hodfunc_sat2=None,
        A=-1,
        epsilon=0,
        flens1=1,
        flens2=1,
    ):
        self.set_hod1(hodfunc_cen1, hodfunc_sat1)
        if hodfunc_cen2 != None:
            self.set_hod2(hodfunc_cen2, hodfunc_sat2)
        else:
            self.set_hod2(hodfunc_cen1, hodfunc_sat1)

        self.set_hod_corr(A, epsilon)

        self.flens1 = flens1
        self.flens2 = flens2

    def u_NFW(self, k, m, z, f):
        a = 1 / (1 + z)
        c = f * self.cm(self.cosmo, m, a)

        r200 = np.power(0.239 * m / 200 / ccl.rho_x(self.cosmo, a, "matter"), 1.0 / 3.0)

        arg1 = k * r200 / c
        arg2 = arg1 * (1 + c)

        si1, ci1 = sici(arg1)
        si2, ci2 = sici(arg2)

        term1 = np.sin(arg1) * (si2 - si1)
        term2 = np.cos(arg1) * (ci2 - ci1)
        term3 = -np.sin(arg1 * c) / arg2

        F = np.log(1.0 + c) - c / (1.0 + c)

        return (term1 + term2 + term3) / F

    def G_ab(self, k1, k2, m, z, a=1, b=2):
        if a==1:
            Nc1=self.hod_cen1(m)
            Ns1=self.hod_sat1(m)
            u1=self.u_NFW(k1, m, z, self.flens1)

        else:
            Nc1=self.hod_cen2(m)
            Ns1=self.hod_cen2(m)
            u1=self.u_NFW(k1, m, z, self.flens2)


        if b==1:
            Nc2=self.hod_cen1(m)
            Ns2=self.hod_sat1(m)
            u2=self.u_NFW(k2, m, z, self.flens1)

        else:
            Nc2=self.hod_cen2(m)
            Ns2=self.hod_cen2(m)
            u2=self.u_NFW(k2, m, z, self.flens2)


        if a==b:
            Nss=(Ns1-1)*Ns1
        else:
            Nss=self.hod_satsat(m)

        return Nc1*Ns2*u2+Nc2*Ns1*u1+Nss*u1*u2


    def G_a(self, k, m, z, a=1):
        if a==1:
            Nc=self.hod_cen1(m)
            Ns=self.hod_sat1(m)
            u=self.u_NFW(k, m, z, self.flens1)

        else:
            Nc=self.hod_cen2(m)
            Ns=self.hod_cen2(m)
            u=self.u_NFW(k, m, z, self.flens2)

        return Nc+Ns*u
    

    def get_ave_numberdensity(self, z, type=1):
        if type==1:
            Nc=self.hod_cen1
            Ns=self.hod_sat1

        else:
            Nc=self.hod_cen2
            Ns=self.hod_cen2
        a=1/(1+z)
        kernel=lambda m: (Nc(m)+Ns(m))*self.dndm(m,z)

        return integrate.quad(kernel, self.mmin, self.mmax)[0]

    def lens_lens_ps_1h(self, ks, z, type1=1, type2=2):

        n1=self.get_ave_numberdensity(z, type1)
        if type1==type2:
            n2=n1
        else:
            n2=self.get_ave_numberdensity(z, type2)

        ks=np.array(ks)
        integral=[]

        for k in ks:
            kernel=lambda m: self.G_ab(k, k, m, z, type1, type2)*self.dndm(m,z)

            integral.append(integrate.quad(kernel, self.mmin, self.mmax)[0])
        #integral=quad(kernel, self.mmin, self.mmax)

        integral=np.array(integral)
        return integral/n1/n2

    # def lens_lens_ps_2h

    # def lens_lens_ps

    # def source_lens_ps_1h

    # def source_lens_ps_2h

    # def source_lens_ps

    # def source_lens_lens_bs_1h

    # def source_lens_lens_bs_2h

    # def source_lens_lens_bs_3h

    # def source_lens_lens_bs

    # def source_source_lens_bs_1h

    # def source_source_lens_bs_2h

    # def source_source_lens_bs_3h

    # def source_source_lens_bs
