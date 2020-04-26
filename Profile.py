import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

G = 4.492*10**(-6) #kpc^3/Modot/Gyr^2

class Profile:

    def __init__(self, trunc_r = np.inf,name=None,**kwargs):
        """
        Profile constructor
        
        Defines basic parameters of the profile
        
        Parameters
        -----------
        trunc_r - Truncation radius of the profile. 
                  Defines the halo radius. When inheriting Profile,
                  trunc_r should be the first argument in when calling
                  super().__init__
        name - Name of the profile. Appears in plots and
               when printing a class instance. If None,
               the name is set to the class name.
        **kwargs - Should contain all the free parameters of the model.
                   All model parameters should pass through **kwargs.
                   
        """
        
        self.__dict__.update(kwargs)
        self._nparams = len(kwargs.keys())
        self.trunc_r = trunc_r
        if name is None:
            self._name = self.__class__.__name__
        else:
            self._name = name
        try:
            self.density(1)
        except Exception as error:
            raise error
    
    def __str__(self):
        output = self._name + ': '
        for arg in self.__dict__:
            if arg == 'trunc_r' or arg[0] == '_':
                continue
            output += arg+' = {}, '.format(self.__dict__[arg])
        return output
    def __repr__(self):
        return self.__str__()

    def density(self,r):
        """
        The spherical density profile of the model.
        This method must be implemented (as it is the core of the class).
        When overridden, should accept only 'self' and 'r'.
        
        Parameters
        -----------
        r - The radius at which the profile is to be evaluated
        """
        raise NotImplementedError("Must implement the profile's density")

    def mass(self,r,r_min=0.,quad_output = False,**kwargs):
        """
        The mass profile of the model.
        If not overriden, returns the numerically integrated
        mass inside 'r'. If the model has an analytic
        expression for the mass profile, this should be overriden.
        
        Parameters
        -----------
        r - The radius within to calculate the enclosed mass.
            Must be a scalar or a 1D array.
            If a 1D array, returns a 1D array of the same size
            with the integral evaluated at each radius.
            Also, if 1D array, the array is clipped between 'r_min'
            and 'self.trunc_r'
        
        r_min - The innermost available radius for integration. 
                'r' must be above this value. Must be smaller
                than `self.trunc_r`
                
        quad_output - Whether or not return the output of quad
                      according to full_output quad argument.
                      Relevant for scalar 'r' only
        
        **kwargs - additional kwargs to pass to quad
        
        Returns
        ________
        quad_result - If quad_output is False, returns the mass 
                      enclosed in 'r' if 'r' is a scalar, or 1D array
                      of the size of 'r' containing the cumulative mass profile.
                      If quad_output is True and 'r' is a scalar,
                      returns the output of quad, with the mass enclosed
                      as the first entry                      
        
        """
        
        if r_min >= self.trunc_r:
            raise ValueError("'r_min' must be within the truncation radius")
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


    def average_density(self,r,r_min=0.,quad_output=False,**kwargs):
        """
        The average density profile.
        This basically returns self.mass / (4*np.pi*r**3/3)
        
        Parameters
        -----------
        r - The radius within to calculate the enclosed mass.
            Must be a scalar or a 1D array.
            If a 1D array, returns a 1D array of the same size
            with the integral evaluated at each radius.
            Also, if 1D array, the array is clipped between 'r_min'
            and 'self.trunc_r'
        
        r_min - The innermost available radius for integration. 
                'r' must be above this value. Must be smaller
                than `self.trunc_r`
                
        quad_output - Whether or not return the output of quad
                      according to full_output quad argument.
                      Relevant for scalar 'r' only
        
        **kwargs - additional kwargs to pass to quad
        
        Returns
        ________
        quad_result - If quad_output is False, returns the mass 
                      enclosed in 'r' if 'r' is a scalar, or 1D array
                      of the size of 'r' containing the cumulative mass profile.
                      If quad_output is True and 'r' is a scalar,
                      returns the output of quad, with the mass enclosed
                      as the first entry                      
        
        """
        
        
        
        try:
            mass_enclosed = self.mass(r,r_min,quad_output,**kwargs)
        except Exception as error:
            raise error
        
        return mass_enclosed / (4*np.pi*r**3/3)
    

    def fit_profile(self,rdata,data,p0=None,logfit=True,new_name=None,**kwargs):
        """
        Fits the density profile to data.
        
        The parameters in the p0 initial guess tuple should be ordered
        in the exact same way they were passed to the constructor, after trunc_r.
        E.g. if the constructor is __init__(self,trunc_r,rho,rs), then p0 should be
        p0=(initial rho, initial rs)
        
        Parameters
        -----------
        rdata - 1D array of points.
        
        data - 1D array of data points of the same size as rdata.
               It is assumed that the fit is for data = density(rdata).
               
        p0 - Initial guess tuple passed to curve_fit. See above about ordering.
        
        logfit - Whether to fit self.density or np.log10(self.density)
        
        new_name - Name of the returned profile. If None, appends ' fit'
                   to the name of the calling profile.
                   
        **kwargs - Additional arguments to pass to curve_fit
        
        Returns
        --------
        fitted_profile - An instance of the same class as the calling profile
                         with the best fit parameters.
        
        params - The best-fit parameters
        
        pcov - The estimated covariance of params. To compute the errors
                for the parameter, use perr = np.sqrt(np.diag(pcov))
        """
        
        if rdata.size != data.size:
            raise ValueError("'rdata' and 'data' must be of the same size")
        
        if logfit:
            fit_func = lambda r,*p: np.log10(self.__class__(self.trunc_r,*p,name=None).density(r))
        else:
            fit_func = lambda r,*p: self.__class__(self.trunc_r,*p,name=None).density(r)
              

        params,pcov = curve_fit(fit_func,rdata,data,p0=p0,**kwargs)
        new_name = self._name+' fit' if new_name is None else new_name
        fitted_profile = self.__class__(self.trunc_r,*params,name=new_name)
        return fitted_profile,params,pcov




    def plot(self,r_min=0., r_points=100,xscale='log',yscale='log',
                 ax=None,r_max=None,legend=True,normalized=True,
                 logspaced=True,**kwargs):
        """
        Plot the density of the calling model between r_min and self.trunc_r.
        If self.trunc_r = np.inf, r_max should be defined.
        
        Parameters
        -----------
        r_min - The minimum radius in the plotting range.
        
        r_points - The number of points to evaluate self.density
        
        xscale - The scale of the x-axis. Should be compatible with
                matplotlib scales.
                
        yscale - The scale of the y-axis. Should be compatible with
                matplotlib scales.
                
        ax - Plot in an existing axis. If None, create new figure.
        
        r_max - Maximum radius in the plotting range. Should be
                <= self.trunc_r if the latter is finite.
                
        legend - Whether or not spawn a legend in ax
        
        normalized - Whether or not x-axis is normalized by trunc_r
        
        logspaced - Whether or not the x-axis is log spaced.
                
        **kwargs - Additional arguments to pass to ax.plot
        
        Returns
        --------
        Returns fig,ax objects if ax=None
                
        """
        
        
        if r_max is None:
            if self.trunc_r == np.inf:
                raise ValueError("'r_max' must be provided for untruncated profiles")
            else:
                r_max = self.trunc_r
        else:
            r_max = np.min([r_max,self.trunc_r])
        
        if ax is None:
            fig,ax1 = plt.subplots(1,1,figsize=(5,5))
        else:
            ax1 = ax
        
        if logspaced:
            rlist = np.logspace(np.log10(r_min),np.log10(r_max),r_points)
        else:
            rlist = np.linspace(r_min,r_max,r_points)
        myname = self.name
        data = self.density(rlist)
        rlist = rlist/self.trunc_r if normalized else r
        ax1.plot(rlist,data,label=myname,**kwargs)
        ax1.set_yscale(yscale)
        ax1.set_xscale(xscale)
        if normalized: #Assuming Rv = trunc_r..
            ax1.set_xlabel(r'$r/Rv$',fontsize=15)
        else:
            ax1.set_xlabel(r'$r\ [\mathrm{kpc}]$',fontsize=15)
        ax1.set_ylabel(r'$\rho$',fontsize=15)
        if legend:
            ax1.legend(loc='best')
            
        if ax is None:
            return fig,ax1


    @staticmethod
    def canonicalToModelParams():
        """ 
        A method to transform `canonical parameters`,
        such as Rv,Mv, concentration, slope etc.
        to model-specific parameters such as rho_0, r_s etc.
        """
        raise NotImplementedError
        
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self,name):
        self._name = name
