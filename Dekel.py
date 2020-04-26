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
    
    @staticmethod
    def mu(c,a):
        return c**(a-3)*(1+c**(1/2))**(2*(3-a))
    @staticmethod
    def rho_vir(Mv,Rv):
        return 3*Mv/(4*np.pi*Rv**3)
    @staticmethod
    def rho_c(Mv,Rv,c,a):
        mu = Dekel.mu(c,a)
        rho_vir = Dekel.rho_vir(Mv,Rv)
        return c**3*mu*rho_vir
    @staticmethod
    def rho(a,rho_c):
        return (1-a/3)*rho_c
    
    @staticmethod
    def getModelParams(Mv,Rv,c,a):
        rho_c = Dekel.rho_c(Mv,Rv,c,a)
        rho = Dekel.rho(a,rho_c)
        return rho,Rv/c,a
        