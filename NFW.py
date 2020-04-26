import numpy as np
from Profile import Profile

class NFW(Profile):
    
    def __init__(self,trunc_r,rho,rs,name=None):
        super().__init__(trunc_r,rho=rho,rs=rs,name=name)
    
    def density(self,r):
        return self.rho/((r/self.rs)*(1+r/self.rs)**2)
    def mass(self,r):
        ndim = np.ndim(r)
        if ndim == 0:
            r = r if r <= self.trunc_r else self.trunc_r
        elif ndim == 1:
            r = np.clip(r,0,self.trunc_r)
        else:
            raise TypeError("'r' must be a scalar or a 1D array")
            
        return 4*np.pi*self.rho*self.rs**3*(np.log((self.rs+r)/self.rs) - r/(r+self.rs))
    
    @staticmethod
    def canonicalToModelParams(Mv,Rv,c):
        return Mv/(4*np.pi*Rv**3/c**3 * (np.log(1+c)-c/(1+c))), Rv/c