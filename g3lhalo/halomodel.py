import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sici
import scipy.integrate as integrate

c_over_H0 = 3000  # Mpc/h


class halomodel:

    def __init__(
        self, verbose=False, cosmo=None, hmfunc=None, hbfunc=None, cmfunc=None
    ):
        """Halomodel initialization. Automatically assigns cosmology, halo mass function, halo bias and concetration mass relation if these are specified.

        Args:
            verbose (bool, optional): Control verbosity. Defaults to False.
            cosmo (dict, optional): Cosmological parameters as dictionary. Needs to include 'Om_b', 'Om_c', 'h', 'sigma_8', 'n_s'. Other parameters are ignored. Defaults to None.
            hmfunc (function, optional): Halo mass function [dn/dlog10(M/Msun)]. Needs to be function of halo mass and scale factor. Defaults to None.
            hbfunc (function, optional): Halo bias function. Needs to be function of halo mass and scale factor. Defaults to None.
            cmfunc (function, optional): Concentration mass relation. Needs to be function of halo mass and scale factor. Defaults to None.
        """

        self.verbose = verbose

        if type(cosmo) is dict:
            self.set_cosmo(cosmo)

        if callable(hmfunc):
            self.set_hmf(hmfunc)

        if callable(hbfunc):
            self.set_halobias(hbfunc)

        if callable(cmfunc):
            self.set_concentration_mass_relation(cmfunc)

    def set_cosmo(self, cosmo):
        """Sets cosmology

        Args:
            cosmo (dict): Cosmological parameters as dictionary. Needs to include 'Om_b', 'Om_c', 'h', 'sigma_8', 'n_s'. Other parameters are ignored.
        """

        if self.verbose:
            print("Setting cosmology")
            print(f"Om_c: {cosmo['Om_c']}")
            print(f"Om_b: {cosmo['Om_b']}")
            print(f"h: {cosmo['h']}")
            print(f"sigma_8: {cosmo['sigma_8']}")
            print(f"n_s: {cosmo['n_s']}")

        self.cosmo = ccl.Cosmology(
            Omega_c=cosmo["Om_c"],
            Omega_b=cosmo["Om_b"],
            h=cosmo["h"],
            sigma8=cosmo["sigma_8"],
            n_s=cosmo["n_s"],
        )

        if self.verbose:
            print("Also setting linear matter power spectrum")

        self.pk_lin = lambda k, z: ccl.linear_matter_power(
            self.cosmo, k, 1.0 / (1.0 + z)
        )

    def set_hmf(self, hmfunc):
        """Sets halo mass function. Note that the internal model.dndm is in units of 1/Msun/Mpc³ and a function of mass and redshift not scale factor!

        Args:
            hmfunc (function): Halo mass function [dn/dlog10(M/Msun)]. Needs to be function of halo mass and scale factor.
        """

        if self.verbose:
            print("Setting halo mass function")
            print(hmfunc)

        self.dndm = lambda m, z: hmfunc(self.cosmo, m, 1.0 / (1.0 + z)) / (
            m * np.log(10)
        )

    def set_halobias(self, hbfunc):
        """Sets halo bias function.

        Args:
            hbfunc (_type_): _description_
        """
        if self.verbose:
            print("Setting halo bias function")
            print(hbfunc)
        self.bh = lambda m, z: hbfunc(self.cosmo, m, 1.0 / (1.0 + z))

    def set_concentration_mass_relation(self, cmfunc):
        if self.verbose:
            print("Setting concentration mass relation")
            print(cmfunc)
        self.cm = cmfunc

    def set_hod1(self, hodfunc_cen, hodfunc_sat):
        self.hod_cen1 = hodfunc_cen
        self.hod_sat1 = hodfunc_sat

    def set_hod2(self, hodfunc_cen, hodfunc_sat):
        self.hod_cen2 = hodfunc_cen
        self.hod_sat2 = hodfunc_sat

    def set_hod_corr(self):
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

    def u_NFW(self, k, m, z, f=1.0):
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

    def get_ave_numberdensity(self, z, type=1, mmin=1e10, mmax=1e17):
        if type == 1:
            Nc = self.hod_cen1
            Ns = self.hod_sat1

        else:
            Nc = self.hod_cen2
            Ns = self.hod_cen2
        a = 1 / (1 + z)
        kernel = lambda m: (Nc(m) + Ns(m)) * self.dndm(m, z)

        return integrate.quad(kernel, mmin, mmax)[0]

    def lens_lens_ps_1h(self, ks, z, type1=1, type2=2, mmin=1e10, mmax=1e17):

        ks = np.array(ks)
        integral = []

        for k in ks:
            kernel = lambda m: self.G_ab(k, k, m, z, type1, type2) * self.dndm(m, z)

            integral.append(integrate.quad(kernel, mmin, mmax)[0])

        integral = np.array(integral)
        return integral

    def lens_lens_ps_2h(self, ks, z, type1=1, type2=2, mmin=1e10, mmax=1e17):

        ks = np.array(ks)

        integral1 = []

        for k in ks:
            kernel = (
                lambda m: self.G_a(k, m, z, type1) * self.dndm(m, z) * self.bh(m, z)
            )
            integral1.append(integrate.quad(kernel, mmin, mmax)[0])

        integral1 = np.array(integral1)

        if type1 == type2:
            integral2 = integral1
        else:
            integral2 = []

            for k in ks:
                kernel = (
                    lambda m: self.G_a(k, m, z, type2) * self.dndm(m, z) * self.bh(m, z)
                )
                integral2.append(integrate.quad(kernel, mmin, mmax)[0])
            integral2 = np.array(integral2)

        return integral1 * integral2 * self.pk_lin(ks, z)

    def lens_lens_ps(self, ks, z, type1=1, type2=2, mmin=1e10, mmax=1e17):

        n1 = self.get_ave_numberdensity(z, type1)
        if type1 == type2:
            n2 = n1
        else:
            n2 = self.get_ave_numberdensity(z, type2)

        One_halo = self.lens_lens_ps_1h(ks, z, type1, type2, mmin, mmax) / n1 / n2
        Two_halo = self.lens_lens_ps_2h(ks, z, type1, type2, mmin, mmax) / n1 / n2

        return One_halo, Two_halo, One_halo + Two_halo

    def source_lens_ps_1h(self, ks, z, type=1, mmin=1e10, mmax=1e17):

        ks = np.array(ks)

        integral = []
        for k in ks:
            kernel = (
                lambda m: self.G_a(k, m, z, type)
                * self.dndm(m, z)
                * m
                * self.u_NFW(k, m, z, 1.0)
            )
            integral.append(integrate.quad(kernel, mmin, mmax)[0])

        integral = np.array(integral)
        return integral

    def source_lens_ps_2h(self, ks, z, type=1, mmin=1e10, mmax=1e17):
        ks = np.array(ks)

        integral1 = []
        for k in ks:
            kernel = lambda m: self.G_a(k, m, z, type) * self.dndm(m, z) * self.bh(m, z)
            integral1.append(integrate.quad(kernel, mmin, mmax)[0])
        integral1 = np.array(integral1)

        integral2 = []
        for k in ks:
            kernel = lambda m: self.u_NFW(k, m, z, 1.0) * self.dndm(m, z) * m
            integral2.append(integrate.quad(kernel, mmin, mmax)[0])
        integral2 = np.array(integral2)

        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")
        corr2 = (rho_bar - self.A_2h_correction(z, mmin, mmax)) * self.u_NFW(
            ks, mmin, z, f=1.0
        )
        integral2 += corr2

        return integral1 * integral2 * self.pk_lin(ks, z)

    def source_lens_ps(self, ks, z, type=1, mmin=1e10, mmax=1e17):
        n = self.get_ave_numberdensity(z, type)
        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")

        One_halo = self.source_lens_ps_1h(ks, z, type, mmin, mmax) / n / rho_bar
        Two_halo = self.source_lens_ps_2h(ks, z, type, mmin, mmax) / n / rho_bar

        return One_halo, Two_halo, One_halo + Two_halo

    def source_source_ps_1h(self, ks, z, mmin=1e10, mmax=1e17):

        ks = np.array(ks)

        integral = []
        for k in ks:
            kernel = lambda m: self.dndm(m, z) * m * m * (self.u_NFW(k, m, z)) ** 2
            integral.append(integrate.quad(kernel, mmin, mmax, limit=100)[0])

        integral = np.array(integral)
        return integral

    def source_source_ps_2h(self, ks, z, mmin=1e10, mmax=1e17):
        ks = np.array(ks)

        integral = []

        for k in ks:
            kernel = lambda m: self.u_NFW(k, m, z) * self.dndm(m, z) * m * self.bh(m, z)
            integral.append(integrate.quad(kernel, mmin, mmax)[0])

        integral = np.array(integral)
        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")

        corr = (rho_bar - self.A_2h_correction(z, mmin, mmax)) * self.u_NFW(ks, mmin, z)
        integral += corr

        return integral**2 * self.pk_lin(ks, z)

    def A_2h_correction(self, z, mmin=1e10, mmax=1e17):
        kernel_A = lambda m: self.dndm(m, z) * self.bh(m, z) * m
        A = integrate.quad(kernel_A, mmin, mmax)[0]
        return A

    def source_source_ps(self, ks, z):
        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")
        One_halo = self.source_source_ps_1h(ks, z) / rho_bar**2
        Two_halo = self.source_source_ps_2h(ks, z) / rho_bar**2

        return One_halo, Two_halo, One_halo + Two_halo

    def source_lens_lens_bs_1h(
        self, k1, k2, k3, z, type1=1, type2=2, mmin=1e10, mmax=1e17
    ):

        kernel = (
            lambda m: self.dndm(m, z)
            * m
            * self.u_NFW(k1, m, z, 1.0)
            * self.G_ab(k2, k3, m, z, type1, type2)
        )
        integral = integrate.quad(kernel, mmin, mmax)[0]

        return integral

    def source_lens_lens_bs_2h(
        self, k1, k2, k3, z, type1=1, type2=2, mmin=1e10, mmax=1e17
    ):

        kernel = (
            lambda m: self.dndm(m, z) * self.u_NFW(k1, m, z, 1.0) * m * self.bh(m, z)
        )
        summand1 = integrate.quad(kernel, mmin, mmax)[0]

        kernel = (
            lambda m: self.dndm(m, z)
            * self.G_ab(k2, k3, m, z, type1, type2)
            * self.bh(m, z)
        )
        summand1 *= integrate.quad(kernel, mmin, mmax)[0]

        summand1 *= self.pk_lin(k3, z)

        kernel = lambda m: self.dndm(m, z) * self.G_a(k2, m, z, type1) * self.bh(m, z)
        summand2 = integrate.quad(kernel, mmin, mmax)[0]

        kernel = (
            lambda m: self.dndm(m, z)
            * self.G_a(k3, m, z, type2)
            * m
            * self.u_NFW(k1, m, z, 1.0)
            * self.bh(m, z)
        )
        summand2 *= integrate.quad(kernel, mmin, mmax)[0]

        summand2 *= self.pk_lin(k1, z)

        kernel = lambda m: self.dndm(m, z) * self.G_a(k3, m, z, type2) * self.bh(m, z)
        summand3 = integrate.quad(kernel, mmin, mmax)[0]

        kernel = (
            lambda m: self.dndm(m, z)
            * self.G_a(k2, m, z, type1)
            * m
            * self.u_NFW(k1, m, z, 1.0)
            * self.bh(m, z)
        )
        summand3 *= integrate.quad(kernel, mmin, mmax)[0]

        summand3 *= self.pk_lin(k2, z)

        return summand1 + summand2 + summand3

    def source_lens_lens_bs_3h(
        self, k1, k2, k3, z, type1=1, type2=2, mmin=1e10, mmax=1e17
    ):

        kernel = lambda m: self.dndm(m, z) * self.G_a(k2, m, z, type1) * self.bh(m, z)
        integral = integrate.quad(kernel, mmin, mmax)[0]

        kernel = lambda m: self.dndm(m, z) * self.G_a(k3, m, z, type2) * self.bh(m, z)
        integral *= integrate.quad(kernel, mmin, mmax)[0]

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k1, m, z, f=1.0)
        integral *= integrate.quad(kernel, mmin, mmax)[0]

        return integral * self.bk_lin(k1, k2, k3, z)

    def source_lens_lens_bs(
        self, k1, k2, k3, z, type1=1, type2=2, mmin=1e10, mmax=1e17
    ):

        n1 = self.get_ave_numberdensity(z, type1)
        if type1 == type2:
            n2 = n1
        else:
            n2 = self.get_ave_numberdensity(z, type2)

        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")

        One_halo = (
            self.source_lens_lens_bs_1h(k1, k2, k3, z, type1, type2, mmin, mmax)
            / n1
            / n2
            / rho_bar
        )

        Two_halo = (
            self.source_lens_lens_bs_2h(k1, k2, k3, z, type1, type2, mmin, mmax)
            / n1
            / n2
            / rho_bar
        )

        Three_halo = (
            self.source_lens_lens_bs_3h(k1, k2, k3, z, type1, type2, mmin, mmax)
            / n1
            / n2
            / rho_bar
        )

        return One_halo, Two_halo, Three_halo, One_halo + Two_halo + Three_halo

    def source_source_lens_bs_1h(self, k1, k2, k3, z, type=1, mmin=1e10, mmax=1e17):

        kernel = (
            lambda m: self.dndm(m, z)
            * self.G_a(k3, m, z, type)
            * m
            * m
            * self.u_NFW(k1, m, z)
            * self.u_NFW(k2, m, z)
        )
        integral = integrate.quad(kernel, mmin, mmax)[0]

        return integral

    def source_source_lens_bs_2h(self, k1, k2, k3, z, type=1, mmin=1e10, mmax=1e17):

        kernel = lambda m: self.dndm(m, z) * self.G_a(k3, m, type) * self.bh(m, z)
        summand1 = integrate.quad(kernel, mmin, mmax)[0]

        kernel = (
            lambda m: self.dndm(m, z)
            * m
            * m
            * self.u_NFW(k1, m, z)
            * self.u_NFW(k2, m, z)
            * self.bh(m, z)
        )
        summand1 *= integrate.quad(kernel, mmin, mmax)[0]

        summand1 *= self.pk_lin(k3, z)

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k1, m, z) * self.bh(m, z)
        summand2 = integrate.quad(kernel, mmin, mmax)[0]

        kernel = (
            lambda m: self.dndm(m, z)
            * m
            * self.u_NFW(k2, m, z)
            * self.G_a(k3, m, z, type)
            * self.bh(m, z)
        )
        summand2 *= integrate.quad(kernel, mmin, mmax)[0]

        summand2 *= self.pk_lin(k1, z)

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k2, m) * self.bh(m, z)
        summand3 = integrate.quad(kernel, mmin, mmax)[0]

        kernel = (
            lambda m: self.dndm(m, z)
            * m
            * self.u_NFW(k1, m, z)
            * self.G_a(k3, m, z, type)
            * self.bh(m, z)
        )

        summand3 *= integrate.quad(kernel, mmin, mmax)[0]

        summand3 *= self.pk_lin(k2, z)

        return summand1 + summand2 + summand3

    def source_source_lens_bs_3h(self, k1, k2, k3, z, type=1, mmin=1e10, mmax=1e17):
        kernel = lambda m: self.dndm(m, z) * self.G_a(k3, m, type) * self.bh(m, z)
        integral = integrate.quad(kernel, mmin, mmax)[0]

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k1, m) * self.bh(m, z)
        integral *= integrate.quad(kernel, mmin, mmax)[0]

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k2, m) * self.bh(m, z)
        integral *= integrate.quad(kernel, mmin, mmax)[0]

        return integral * self.bk_lin(k1, k2, k3, z)

    def source_source_lens_bs(self, k1, k2, k3, z, type=1, mmin=1e10, mmax=1e17):
        n = self.get_ave_numberdensity(z, type)
        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")

        One_halo = (
            self.source_source_lens_bs_1h(k1, k2, k3, z, type, mmin, mmax)
            / n
            / rho_bar
            / rho_bar
        )

        Two_halo = (
            self.source_source_lens_bs_2h(k1, k2, k3, z, type, mmin, mmax)
            / n
            / rho_bar
            / rho_bar
        )

        Three_halo = (
            self.source_source_lens_bs_3h(k1, k2, k3, z, type, mmin, mmax)
            / n
            / rho_bar
            / rho_bar
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
