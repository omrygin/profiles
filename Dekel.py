"""
Implementation of the Dekel+ profile (Dekel et al. 2017, Freundlich et al 2020a,b)

"""

import numpy as np
from Profile import Profile

class Dekel(Profile):
    
    def __init__(self,trunc_r,rho,rs,a,name=None):
        super().__init__(trunc_r,rho=rho,rs=rs,a=a,name=name)
    
    def density(self,r):
        x = r/self.rs
        a = self.a
        
        return self.rho / (x**a*(1+x**(1/2))**(2*(3.5-a)))
    
    def average_density(self,r):
        rho_c_bar = self.rho/(1-self.a/3)
        x = r/self.rs
        a = self.a
        return rho_c_bar / (x**a*(1+x**(1/2))**(2*(3.-a)))
    
    def mass(self,r):
        return 4*np.pi*r**3 * self.average_density(r)