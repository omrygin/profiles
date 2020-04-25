import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

G = 4.492*10**(-6) #kpc^3/Modot/Gyr^2

class Profile:

    def __init__(self, trunc_r = np.inf,*args,**kwargs):
        self.__dict__.update(kwargs)
        self.trunc_r = trunc_r

    def density(self,r):
        raise NotImplementedError("Must implement the profile's density")

    def mass(self,r,r_min=0.,quad_output = False,*args,**kwargs):
        ndim = np.ndim(r)
        if ndim <= 1:
            point_eval = True if ndim == 0 else False
            r = r if ndim == 0 else np.array(r)
        else:
            raise TypeError("'r' must be a scalar or a 1D array")

        if point_eval:
            if r < r_min:
                raise ValueError("Cannot integrate below the minimum radius")
            if r_min > self.trunc_r:
                raise ValueError("Lower bounds of integration must be inside"
                        " the truncation radius")

            if r > self.trunc_r:
                r = self.trunc_r
        else:
            if r.max() < r_min or r.min() > self.trunc_r:
                raise ValueError("The values of 'r' must be between"
                        " r_min and the truncation radius")
            r = np.clip(r,r_min,self.trunc_r)
        if point_eval:
            if quad_output:
                quad_result = quad(lambda x:4*np.pi*self.density(x)*x**2,r_min,r,**kwargs)
            else:
                quad_result = quad(lambda x:4*np.pi*self.density(x)*x**2,r_min,r,**kwargs)[0]
        else:
            quad_result = np.zeros_like(r)
            for ir,rad in enumerate(r):
                quad_result[ir] = quad(lambda x:4*np.pi*self.density(x)*x**2,r_min,rad,**kwargs)[0]

        return quad_result




    def compare_with(self,other,r_min=0.,r_points=100,xscale='log',yscale='log'):
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        rlist = np.linspace(r_min,self.trunc_r,r_points)
        myname = self.__class__.__name__
        othername = other.__class__.__name__
        ax.plot(rlist,self.density(rlist),label=myname)
        ax.plot(rlist,other.density(rlist),label=othername)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.legend(loc='best')


