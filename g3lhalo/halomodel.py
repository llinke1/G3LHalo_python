import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sici
import scipy.integrate as integrate

# from quadpy import quad

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

        # self.pk_lin = ccl.linear_matter_power(self.cosmo, ks, a)
        self.pk_lin = lambda k, z: ccl.linear_matter_power(
            self.cosmo, k, 1.0 / (1.0 + z)
        )

        # self.w = ccl.comoving_angular_distance(self.cosmo, a)
        # self.dwdz = 1 / np.sqrt(ccl.h_over_h0(self.cosmo, a)) * c_over_H0



    # def set_nz_sources(self, zs, n):
    #     zs_new = np.linspace(self.zmin, self.zmax, self.nbins)
    #     self.n_s = np.interp(zs_new, zs, n, left=0, right=0)
    #     self.set_lensing_efficiency()

    # def set_lensing_efficiency(self):
    #     self.gs = np.zeros_like(self.w)
    #     dz = (self.zmax - self.zmin) / self.nbins

    #     for i, w in enumerate(self.w):
    #         for j, wprime in enumerate(self.w):
    #             if wprime >= w:
    #                 self.gs[i] += self.n_s[j] * (wprime - w) / wprime * dz

    def set_hmf(self, hmfunc):
        # ms = np.geomspace(self.mmin, self.mmax, self.nbins)
        # zs = np.linspace(self.zmin, self.zmax, self.nbins)
        # a = 1 / (1 + zs)

        # self.dndm = []
        # for a_ in a:
        #    self.dndm.append(hmfunc(self.cosmo, ms, a_) / (ms * np.log(10)))

        # self.dndm = np.array(self.dndm)
        self.dndm = lambda m, z: hmfunc(self.cosmo, m, 1.0 / (1.0 + z)) / (
            m * np.log(10)
        )

    def set_halobias(self, hbfunc):
        self.bh = lambda m, z: hbfunc(self.cosmo, m, 1.0 / (1.0 + z))
        # ms = np.geomspace(self.mmin, self.mmax, self.nbins)
        # zs = np.linspace(self.zmin, self.zmax, self.nbins)
        # a = 1 / (1 + zs)

        # self.bh = []
        # for a_ in a:
        #     self.bh.append(hbfunc(self.cosmo, ms, a_))

        # self.bh = np.array(self.bh)

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

    def set_hod_corr(self):
        # ms = np.geomspace(self.mmin, self.mmax, self.nbins)
        # self.hod_satsat = self.hod_sat1 * self.hod_sat2 + A * np.power(
        #     ms, epsilon
        # ) * np.sqrt(self.hod_sat1 * self.hod_sat2)

        self.hod_satsat = lambda m: self.hod_sat1(m) * self.hod_sat2(
            m
        ) + self.A * np.power(m, self.epsilon) * np.sqrt(
            self.hod_sat1(m) * self.hod_sat2(m)
        )

    def set_hods(
        self,
        hodfunc_cen1,
        hodfunc_sat1,
        hodfunc_cen2=None,
        hodfunc_sat2=None,
        A=0,
        epsilon=0,
        flens1=1,
        flens2=1,
    ):
        self.set_hod1(hodfunc_cen1, hodfunc_sat1)
        if hodfunc_cen2 != None:
            self.set_hod2(hodfunc_cen2, hodfunc_sat2)
        else:
            self.set_hod2(hodfunc_cen1, hodfunc_sat1)

        self.A = A
        self.epsilon = epsilon
        self.set_hod_corr()

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
        if a == 1:
            Nc1 = self.hod_cen1(m)
            Ns1 = self.hod_sat1(m)
            u1 = self.u_NFW(k1, m, z, self.flens1)

        else:
            Nc1 = self.hod_cen2(m)
            Ns1 = self.hod_cen2(m)
            u1 = self.u_NFW(k1, m, z, self.flens2)

        if b == 1:
            Nc2 = self.hod_cen1(m)
            Ns2 = self.hod_sat1(m)
            u2 = self.u_NFW(k2, m, z, self.flens1)

        else:
            Nc2 = self.hod_cen2(m)
            Ns2 = self.hod_cen2(m)
            u2 = self.u_NFW(k2, m, z, self.flens2)

        if a == b:
            Nss = (Ns1 - 1) * Ns1
        else:
            Nss = self.hod_satsat(m)

        return Nc1 * Ns2 * u2 + Nc2 * Ns1 * u1 + Nss * u1 * u2

    def G_a(self, k, m, z, a=1):
        if a == 1:
            Nc = self.hod_cen1(m)
            Ns = self.hod_sat1(m)
            u = self.u_NFW(k, m, z, self.flens1)

        else:
            Nc = self.hod_cen2(m)
            Ns = self.hod_cen2(m)
            u = self.u_NFW(k, m, z, self.flens2)

        return Nc + Ns * u

    def get_ave_numberdensity(self, z, type=1):
        if type == 1:
            Nc = self.hod_cen1
            Ns = self.hod_sat1

        else:
            Nc = self.hod_cen2
            Ns = self.hod_cen2
        a = 1 / (1 + z)
        kernel = lambda m: (Nc(m) + Ns(m)) * self.dndm(m, z)

        return integrate.quad(kernel, self.mmin, self.mmax)[0]

    def get_ave_matterdensity(self, z):
        kernel = lambda m: self.dndm(m, z) * m

        return integrate.quad(kernel, self.mmin, self.mmax)[0]

    def lens_lens_ps_1h(self, ks, z, type1=1, type2=2):

        ks = np.array(ks)
        integral = []

        for k in ks:
            kernel = lambda m: self.G_ab(k, k, m, z, type1, type2) * self.dndm(m, z)

            integral.append(integrate.quad(kernel, self.mmin, self.mmax)[0])
        # integral=quad(kernel, self.mmin, self.mmax)

        integral = np.array(integral)
        return integral

    def lens_lens_ps_2h(self, ks, z, type1=1, type2=2):

        ks = np.array(ks)

        integral1 = []

        for k in ks:
            kernel = (
                lambda m: self.G_a(k, m, z, type1) * self.dndm(m, z) * self.bh(m, z)
            )
            integral1.append(integrate.quad(kernel, self.mmin, self.mmax)[0])

        integral1 = np.array(integral1)

        if type1 == type2:
            integral2 = integral1
        else:
            integral2 = []

            for k in ks:
                kernel = (
                    lambda m: self.G_a(k, m, z, type2) * self.dndm(m, z) * self.bh(m, z)
                )
                integral2.append(integrate.quad(kernel, self.mmin, self.mmax)[0])
            integral2 = np.array(integral2)

        return integral1 * integral2 * self.pk_lin(ks, z)

    def lens_lens_ps(self, ks, z, type1=1, type2=2):

        n1 = self.get_ave_numberdensity(z, type1)
        if type1 == type2:
            n2 = n1
        else:
            n2 = self.get_ave_numberdensity(z, type2)

        One_halo = self.lens_lens_ps_1h(ks, z, type1, type2) / n1 / n2
        Two_halo = self.lens_lens_ps_2h(ks, z, type1, type2) / n1 / n2

        return One_halo, Two_halo, One_halo + Two_halo

    def source_lens_ps_1h(self, ks, z, type=1):

        ks = np.array(ks)

        integral = []
        for k in ks:
            kernel = (
                lambda m: self.G_a(k, m, z, type)
                * self.dndm(m, z)
                * m
                * self.u_NFW(k, m, z, 1.0)
            )
            integral.append(integrate.quad(kernel, self.mmin, self.mmax)[0])

        integral = np.array(integral)
        return integral

    def source_lens_ps_2h(self, ks, z, type=1):
        ks = np.array(ks)

        integral1 = []
        for k in ks:
            kernel = lambda m: self.G_a(k, m, z, type) * self.dndm(m, z) * self.bh(m, z)
            integral1.append(integrate.quad(kernel, self.mmin, self.mmax)[0])
        integral1 = np.array(integral1)


        integral2 = []
        for k in ks:
            kernel = lambda m: self.u_NFW(k, m, z, 1.0) * self.dndm(m, z) * m * self.bh(m, z)
            integral2.append(integrate.quad(kernel, self.mmin, self.mmax)[0])
        
        integral2 = np.array(integral2)

        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")
        corr2 = (rho_bar - self.A_2h_correction(ks, z))* self.u_NFW(ks, self.mmin, z, f=1.0)
        integral2 += corr2


        return integral1 * integral2 * self.pk_lin(ks, z)

    def source_lens_ps(self, ks, z, type=1):
        n = self.get_ave_numberdensity(z, type)
        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")

        One_halo = self.source_lens_ps_1h(ks, z, type) / n / rho_bar
        Two_halo = self.source_lens_ps_2h(ks, z, type) / n / rho_bar

        return One_halo, Two_halo, One_halo + Two_halo

    def source_source_ps_1h(self, ks, z):

        ks = np.array(ks)

        integral = []
        for k in ks:
            kernel = lambda m: self.dndm(m, z) * m * m * (self.u_NFW(k, m, z, 1.0)) ** 2
            integral.append(integrate.quad(kernel, self.mmin, self.mmax, limit=100)[0])

        integral = np.array(integral)
        return integral

    def source_source_ps_2h(self, ks, z):
        ks = np.array(ks)

        integral = []

        for k in ks:
            kernel = (
                lambda m: self.u_NFW(k, m, z, 1.0) * self.dndm(m, z) * m * self.bh(m, z)
            )
            integral.append(integrate.quad(kernel, self.mmin, self.mmax)[0])

        integral = np.array(integral)
        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")

        corr = (rho_bar - self.A_2h_correction(ks, z))* self.u_NFW(ks, self.mmin, z, f=1.0)
        integral += corr

        return integral**2 * self.pk_lin(ks, z)

    def A_2h_correction(self, ks, z):
        kernel_A = lambda m: self.dndm(m, z) * self.bh(m, z) * m
        A = integrate.quad(kernel_A, self.mmin, self.mmax)[0] 
        return A

    def source_source_ps(self, ks, z):
        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")
        One_halo = self.source_source_ps_1h(ks, z) / rho_bar**2
        Two_halo = self.source_source_ps_2h(ks, z) / rho_bar**2

        return One_halo, Two_halo, One_halo + Two_halo

    def source_lens_lens_bs_1h(self, k1, k2, k3, z, type1=1, type2=2):

        kernel = (
            lambda m: self.dndm(m, z)
            * m
            * self.u_NFW(k1, m, z, 1.0)
            * self.G_ab(k2, k3, m, z, type1, type2)
        )
        integral = integrate.quad(kernel, self.mmin, self.mmax)[0]

        return integral

    def source_lens_lens_bs_2h(self, k1, k2, k3, z, type1=1, type2=2):

        kernel = (
            lambda m: self.dndm(m, z) * self.u_NFW(k1, m, z, 1.0) * m * self.bh(m, z)
        )
        summand1 = integrate.quad(kernel, self.mmin, self.mmax)[0]

        kernel = (
            lambda m: self.dndm(m, z)
            * self.G_ab(k2, k3, m, z, type1, type2)
            * self.bh(m, z)
        )
        summand1 *= integrate.quad(kernel, self.mmin, self.mmax)[0]

        summand1 *= self.pk_lin(k3, z)

        kernel = lambda m: self.dndm(m, z) * self.G_a(k2, m, z, type1) * self.bh(m, z)
        summand2 = integrate.quad(kernel, self.mmin, self.mmax)[0]

        kernel = (
            lambda m: self.dndm(m, z)
            * self.G_a(k3, m, z, type2)
            * m
            * self.u_NFW(k1, m, z, 1.0)
            * self.bh(m, z)
        )
        summand2 *= integrate.quad(kernel, self.mmin, self.mmax)[0]

        summand2 *= self.pk_lin(k1, z)

        kernel = lambda m: self.dndm(m, z) * self.G_a(k3, m, z, type2) * self.bh(m, z)
        summand3 = integrate.quad(kernel, self.mmin, self.mmax)[0]

        kernel = (
            lambda m: self.dndm(m, z)
            * self.G_a(k2, m, z, type1)
            * m
            * self.u_NFW(k1, m, z, 1.0)
            * self.bh(m, z)
        )
        summand3 *= integrate.quad(kernel, self.mmin, self.mmax)[0]

        summand3 *= self.pk_lin(k2, z)

        return summand1 + summand2 + summand3

    def source_lens_lens_bs_3h(self, k1, k2, k3, z, type1=1, type2=2):

        kernel = lambda m: self.dndm(m, z) * self.G_a(k2, m, z, type1) * self.bh(m, z)
        integral = integrate.quad(kernel, self.mmin, self.mmax)[0]

        kernel = lambda m: self.dndm(m, z) * self.G_a(k3, m, z, type2) * self.bh(m, z)
        integral *= integrate.quad(kernel, self.mmin, self.mmax)[0]

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k1, m, z, f=1.0)
        integral *= integrate.quad(kernel, self.mmin, self.mmax)[0]

        return integral * self.bk_lin(k1, k2, k3, z)

    def source_lens_lens_bs(self, k1, k2, k3, z, type1=1, type2=2):

        n1 = self.get_ave_numberdensity(z, type1)
        if type1 == type2:
            n2 = n1
        else:
            n2 = self.get_ave_numberdensity(z, type2)

        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")

        One_halo = (
            self.source_lens_lens_bs_1h(k1, k2, k3, z, type1, type2) / n1 / n2 / rho_bar
        )

        Two_halo = (
            self.source_lens_lens_bs_2h(k1, k2, k3, z, type1, type2) / n1 / n2 / rho_bar
        )

        Three_halo = (
            self.source_lens_lens_bs_3h(k1, k2, k3, z, type1, type2) / n1 / n2 / rho_bar
        )

        return One_halo, Two_halo, Three_halo, One_halo + Two_halo + Three_halo

    def source_source_lens_bs_1h(self, k1, k2, k3, z, type=1):

        kernel = (
            lambda m: self.dndm(m, z)
            * self.G_a(k3, m, z, type)
            * m
            * m
            * self.u_NFW(k1, m, z, f=1.0)
            * self.u_NFW(k2, m, z, f=1.0)
        )
        integral = integrate.quad(kernel, self.mmin, self.mmax)[0]

        return integral

    def source_source_lens_bs_2h(self, k1, k2, k3, z, type=1):

        kernel = lambda m: self.dndm(m, z) * self.G_a(k3, m, type) * self.bh(m, z)
        summand1 = integrate.quad(kernel, self.mmin, self.mmax)[0]

        kernel = (
            lambda m: self.dndm(m, z)
            * m
            * m
            * self.u_NFW(k1, m, z, 1.0)
            * self.u_NFW(k2, m, z, 1.0)
            * self.bh(m, z)
        )
        summand1 *= integrate.quad(kernel, self.mmin, self.mmax, limit=100)[0]

        summand1 *= self.pk_lin(k3, z)

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k1, m, z, 1.0) * self.bh(m, z)
        summand2 = integrate.quad(kernel, self.mmin, self.mmax)[0]

        kernel = (
            lambda m: self.dndm(m, z)
            * m
            * self.u_NFW(k2, m, z, 1.0)
            * self.G_a(k3, m, z, type)
            * self.bh(m, z)
        )
        summand2 *= integrate.quad(kernel, self.mmin, self.mmax)[0]

        summand2 *= self.pk_lin(k1, z)

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k2, m, z, 1.0) * self.bh(m, z)
        summand3 = integrate.quad(kernel, self.mmin, self.mmax)[0]

        kernel = (
            lambda m: self.dndm(m, z)
            * m
            * self.u_NFW(k1, m, z, 1.0)
            * self.G_a(k3, m, z, type)
            * self.bh(m, z)
        )

        summand3 *= integrate.quad(kernel, self.mmin, self.mmax)[0]

        summand3 *= self.pk_lin(k2, z)

        return summand1 + summand2 + summand3

    def source_source_lens_bs_3h(self, k1, k2, k3, z, type=1):
        kernel = lambda m: self.dndm(m, z) * self.G_a(k3, m,z, type) * self.bh(m, z)
        integral = integrate.quad(kernel, self.mmin, self.mmax)[0]

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k1, m, z, 1.0) * self.bh(m, z)
        integral *= integrate.quad(kernel, self.mmin, self.mmax)[0]

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k2, m, z, 1.0) * self.bh(m, z)
        integral *= integrate.quad(kernel, self.mmin, self.mmax)[0]

        return integral * self.bk_lin(k1, k2, k3, z)

    def source_source_lens_bs(self, k1, k2, k3, z, type=1):
        n = self.get_ave_numberdensity(z, type)
        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")

        One_halo = (
            self.source_source_lens_bs_1h(k1, k2, k3, z, type) / n / rho_bar / rho_bar
        )

        Two_halo = (
            self.source_source_lens_bs_2h(k1, k2, k3, z, type) / n / rho_bar / rho_bar
        )

        Three_halo = (
            self.source_source_lens_bs_3h(k1, k2, k3, z, type) / n / rho_bar / rho_bar
        )

        return One_halo, Two_halo, Three_halo, One_halo + Two_halo + Three_halo

    def bk_lin(self, k1, k2, k3, z):

        c_phi12 = (k3 * k3 - k2 * k2 - k1 * k1) / (2 * k1 * k2)

        c_phi13 = (k2 * k2 - k3 * k3 - k1 * k1) / (2 * k1 * k3)

        c_phi23 = (k1 * k1 - k2 * k2 - k3 * k3) / (2 * k3 * k2)

        return 2 * (
            self.F(k1, k2, c_phi12) * self.pk_lin(k1, z) * self.pk_lin(k2, z)
            + self.F(k1, k3, c_phi13) * self.pk_lin(k1, z) * self.pk_lin(k3, z)
            + self.F(k2, k3, c_phi23) * self.pk_lin(k2, z) * self.pk_lin(k3, z)
        )

    def F(self, k1, k2, cosphi):
        return 0.7143 + 0.2857 * cosphi * cosphi + 0.5 * cosphi * (k1 / k2 + k2 / k2)
