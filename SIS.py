import numpy as np
from Profile import Profile
G = 4.492*10**(-6) #kpc^3/Modot/Gyr^2



class SIS(Profile):

    def __init__(self,trunc_r,Vc):
        super().__init__(trunc_r,Vc=Vc)

    def density(self,r):
        return self.Vc**2/(4*np.pi*G*r**2)

class SIS_analytic(SIS):
    def __init__(self,trunc_r,Vc):
        super().__init__(trunc_r,Vc)

    def mass(self,r):
        ndim = np.ndim(r)
        if ndim == 0:
            r = r if r<=self.trunc_r else self.trunc_r
        elif ndim == 1:
            r = np.clip(r,0,self.trunc_r)
        else:
            raise TypeError("'r' must be a scalar or a 1D array")

        return r*self.Vc**2/G


