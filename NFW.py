import numpy as np
from Profile import Profile


class NFW(Profile):
    
    def __init__(self,trunc_r,rho,rs):
        super().__init__(trunc_r,rho=rho,rs=rs)
    
    def density(self,r):
        return self.rho/((r/self.rs)*(1+r/self.rs)**2)
class NFW_analytic(NFW):
    
    def __init__(self,trunc_r,rho,rs):
        super().__init__(trunc_r,rho,rs)
    
    def mass(self,r):
        ndim = np.ndim(r)
        if ndim == 0:
            r = r if r <= self.trunc_r else self.trunc_r
        elif ndim == 1:
            r = np.clip(r,0,self.trunc_r)
        else:
            raise TypeError("'r' must be a scalar or a 1D array")
            
        return 4*np.pi*self.rho*self.rs**3*(np.log((self.rs+r)/self.rs) - r/(r+self.rs))