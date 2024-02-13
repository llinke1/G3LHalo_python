import numpy as np

class aperturestatistics:

    def __init__():
        pass

    def u_hat(self, eta):
        return 0.5*eta*eta*np.exp(-0.5*eta*eta)
    

    def calculateSourceLensLensBells(self, projSpec, type1, type2, ellMin, ellMax, Nbins):
        self.B_sll=np.zeros((Nbins, Nbins, Nbins))

        ells_edges=np.geomspace(ellMin, ellMax, Nbins+1)
        ells=0.5*ells_edges[1:]+0.5*ells_edges[:-1]
        self.ells=ells
        self.delta_ells=ells_edges[1:]-ells_edges[:-1]

        phis=np.linspace(0, 2*np.pi, Nbins)

        for i, ell1 in enumerate(ells):
            for j, ell2 in enumerate(ells):
                for k, phi in enumerate(phis):
                    ell3=np.sqrt(ell1*ell1+ell2*ell2+2*ell1*ell2*np.cos(phi))

                    self.B_sll[i][j][k] = projSpec.C_kgg(ell1, ell2, ell3, type1, type2)


    def MapNapNap(self, th1, th2, th3, ell):

        result=0

        