import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from netCDF4 import Dataset
from scipy import stats

def km2kstep(l):
    """Convert lenghscales (m) to wavenumbers (cycle/m)"""
    return 1./l

def estimate_slope(pspec,kstep):
    power, intercept, r_value, p_value, std_err = stats.linregress(np.log(kstep),np.log(pspec))
    return power

def prepare_spectrum_plot(pspec,kstep,klmin,klmax):
    _kstep = kstep
    # plotting lines for estimating slopes
    kstepmin = km2kstep(klmin)
    kstepmax = km2kstep(klmax)
    kstep_r = _kstep[(_kstep<kstepmin)*(_kstep>kstepmax)]
    pspec_r = pspec[(_kstep<kstepmin)*(_kstep>kstepmax)]
    kval = estimate_slope(pspec_r,kstep_r)
    kstr = "{:1.1f}".format(kval)
    mpspc = pspec_r.max()
    mkstp = kstep_r[pspec_r == mpspc][0]
    toplt = 10 * mpspc *(kstep_r/kstep_r[0])**(kval) #log to normal value
    return kstr,_kstep,toplt,kstep_r,pspec_r

def plot_spec(kstr,_kstep,toplt,kstep_r,pspec_r,pspec):
    y_min = 10 ** np.floor(np.log10(pspec.min())-1)
    y_max = 10 ** np.ceil( np.log10(pspec.max())+1)
    #-plot figure
    plt.plot(_kstep[0:], pspec[0:],'g-', lw=2)
    plt.plot(kstep_r,  toplt, 'k--', lw=1.5, label=r'$k^{'+kstr+'}$')
    
    logkstp = np.log(kstep_r)
    logpsp = np.log(pspec_r)
    xpos = np.exp((logkstp[0] + logkstp[-1])/2.)
    ypos = np.exp( (logpsp[0] + logpsp[-1])/2.)*12
    
    plt.text(xpos,ypos,kstr,fontsize=9)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('cycle/m',fontsize=12)
    plt.ylim(y_min, y_max)
    plt.legend(fontsize=10)
    plt.grid('on')

def plot_spectrum(pspec,kstep,klmin,klmax):
    kstr,_kstep,toplt,kstep_r,pspec_r = prepare_spectrum_plot(pspec,kstep,klmin,klmax)
    plot_spec(kstr,_kstep,toplt,kstep_r,pspec_r,pspec)

def get_scale(kstep,pspec):
    psd_max = pspec.argmax()
    psd_max_kstep = kstep[psd_max]
    wavelength = 1.0/psd_max_kstep
    scale = wavelength/1E3
    return scale

def get_slope(kstep,pspec,klmin,klmax):
    _kstep = kstep
    kstepmin = km2kstep(klmin)
    kstepmax = km2kstep(klmax)
    kstep_r = _kstep[(_kstep<kstepmin)*(_kstep>kstepmax)]
    pspec_r = pspec[(_kstep<kstepmin)*(_kstep>kstepmax)]
    power, intercept, r_value, p_value, std_err = stats.linregress(np.log(kstep_r),np.log(pspec_r))
    return power
