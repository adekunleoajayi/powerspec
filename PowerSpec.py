############################################################################
# This scripts estimate wavenumber spectrum of two dimensional oceanic dataset such as SSH, vorticity. Prior to
# spectrum estimate, the dataset is interpolated and any NaN value is replaced with interpolated value. A
# tapering is applied to the dataset, afterwhich, the dataset is detrend in both direction. The 2D spectral
# obtained after FFT is radially averaged to a 1D spectral.
#
# Author : Adekunle Ajayi and Julien Lesommer
# Affilation : Institut des Geosciences de l'Environnement (IGE),
#              Universite Grenoble Alpes, France.
# Email : adekunle.ajayi@univ-grenoble-alpes.fr, julien.lesommer@univ-grenoble-alpes.fr
############################################################################


## load modules
import numpy as np
import pandas as pd
import numpy.fft as fft
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import scipy.signal as signal
from scipy.interpolate import interp1d


def e1e2(navlon,navlat):
    """Compute scale factors from navlon,navlat.
        """
    earthrad = 6371229     # mean earth radius (m)
    deg2rad = np.pi / 180.
    lam = navlon
    phi = navlat
    djlam,dilam = np.gradient(lam)
    djphi,diphi = np.gradient(phi)
    e1 = earthrad * deg2rad * np.sqrt( (dilam * np.cos(deg2rad*phi))**2. + diphi**2.)
    e2 = earthrad * deg2rad * np.sqrt( (djlam * np.cos(deg2rad*phi))**2. + djphi**2.)
    return e1,e2


def interpolate(data,navlon,navlat,interp=None):
    """
        interpolate(data,navlon,navlat,interp=None)
        
        Perform a spatial interpolation if required; return x_reg,y_reg,data_reg.
        data : raw data
        nalon : longitude
        navlat : latitude
        interp : if None return data with cordinates in meters, if 'basemap', return interpolated 
        data using basemap from mpl_toolkits and also cordinates in meters.
    """
    e1,e2 = e1e2(navlon,navlat) # ideally we would like e1u and not e1t...
    x1d_in = e1[0,:].cumsum() - e1[0,0]
    y1d_in = e2[:,0].cumsum() - e2[0,0]
    x2d_in,y2d_in = np.meshgrid(x1d_in,y1d_in)
    # print x1d_in
    if interp is None or interp=='0':
        return x2d_in, y2d_in, data.copy()
    elif interp=='basemap': # only for rectangular grid...
        from mpl_toolkits import basemap
        x1d_reg=np.linspace(x1d_in[0],x1d_in[-1],len(x1d_in))
        y1d_reg=np.linspace(y1d_in[0],y1d_in[-1],len(y1d_in))
        x2d_reg,y2d_reg = np.meshgrid(x1d_reg,y1d_reg)
        data_reg=basemap.interp(data,x1d_in,y1d_in,x2d_reg,y2d_reg,checkbounds=False,order=1)
        return x2d_reg,y2d_reg,data_reg


def isdata_contain_nan(data):
    ''' 
        isdata_contain_nan(data)
        
        This function check if a data contains any NaN value
        If yes, it replaces the NaN values with an interpolated value using the fill_nan function.
    '''
    i_mask = data.mask
    arr = np.array(data)
    arr[i_mask] = np.nan
    df = pd.DataFrame(arr)
    if df.isnull().values.any() == True:
        data_new = fill_nan(arr)
        return data_new
    else :
        return data

def fill_nan(data):
    '''replaces a NaN value in a dataset with an interpolated one'''
    # - Reshape to 1D array
    i,j = data.shape
    _1D_arr = data.reshape(i*j,1)
    _1D_arr = _1D_arr.squeeze()
    # - get nan
    nt_nan = np.logical_not(np.isnan(_1D_arr))
    # - get array length
    indice = np.arange(len(_1D_arr))
    # - interpolate for nan value only
    interp = interp1d(indice[nt_nan], _1D_arr[nt_nan],kind ='linear')
    # - replace nan values
    arr = interp(indice)
    # - reshape back to 2D array
    data_nonan = arr.reshape(i,j)
    return data_nonan


def tukey(Ni,Nj):
    '''Using tukey window : tapered cosine window. /alpha = 0.5'''
    wdwi =  signal.tukey(Ni,0.5)
    wdwj =  signal.tukey(Nj,0.5)
    wdw = wdwi[np.newaxis,...]*wdwj[...,np.newaxis]
    return wdw

def hanning(Ni,Nj):
    ''' Using Hanning window'''
    wdwi =  signal.hanning(Ni)
    wdwj =  signal.hanning(Nj)
    wdw = wdwi[np.newaxis,...]*wdwj[...,np.newaxis]
    return wdw

def scaled_han(Ni,Nj):
    ''' Using Hanning window'''
    wdwi =  signal.hanning(Ni)
    wdwi =  (Ni/(wdwi**2).sum())*wdwi
    wdwj =  signal.hanning(Nj)
    wdwj =  (Nj/(wdwj**2).sum())*wdwj
    wdw = wdwi[np.newaxis,...]*wdwj[...,np.newaxis]
    return wdw


def wavenumber_vector(Ni,Nj,dx,dy):
    ''' Compute a wavenumber vector  '''
    kx = np.fft.fftshift(np.fft.fftfreq(Ni,dx)) # two sided
    ky = np.fft.fftshift(np.fft.fftfreq(Nj,dy))
    
    k, l = np.meshgrid(kx,ky)
    kh = np.sqrt(k**2 + l**2)
    kmax = min(k.max(),l.max())
    
    dkx = np.abs(kx[2]-kx[1])
    dky = np.abs(ky[2]-ky[1])
    dkradial = np.sqrt(dkx**2 + dky**2)
    
    # radial wavenumber
    kradial = dkradial*np.arange(1,int(kmax/dkradial))
    return kradial,kh


def get_spec_2D(data_reg,dx,dy,Ni,Nj):
    ''' Compute the 2D spectrum of the data '''
    spec_fft = np.fft.fft2(data_reg)
    spec_2D = (spec_fft*spec_fft.conj()).real*(dx*dy)/(Ni*Nj)
    spec_2D = np.fft.fftshift(spec_2D)
    return spec_2D

def get_spec_1D(kh,kspec,spec_2D):
    ''' Compute the azimuthaly avearge of the 2D spectrum '''
    spec_1D = np.zeros(len(kspec))
    for i in range(kspec.size):
        kr =  (kh>=kspec[i] - kspec[0]) & (kh<=kspec[i]+kspec[0])
        dth = 2.0*np.pi / (kr.sum()-1)
        spec_1D[i] = (spec_2D[kr]*(kh[kr])*dth).sum()
    return spec_1D


def get_spectrum(data_reg,x_reg,y_reg,window='tukey',detrend='both'):
    """ 
        get_spectrum(data_reg,x_reg,y_reg,window='tukey',detrend='both')
        
        data_reg : Interpolated data.
        x_reg and y_reg : interpolate coordinates in meters.
        window : None , 'han' (hanning) or 'tukey' (tappered consine window with /apha = 0.5).
        detrend : 
            if "both" : detrend the 2D data along both axes.
            if "zonal" : detrend the data in the zonal direction only
            if "RemoveMean" : Remove only the mean of the data
            if 'RmeanDtrend' : Remove the mean then detrend the data in both direction
            if None : use the raw data
    """
    Nj,Ni = data_reg.shape
    # detrend
    if detrend is None :
        data_reg = data_reg
    elif detrend == 'both':
        # - detrend data in both direction
        data_reg = signal.detrend(data_reg,axis=0,type='linear')
        data_reg = signal.detrend(data_reg,axis=1,type='linear')
    elif detrend == 'zonal':
        # - detrend data in the zonal direction
        data_reg = signal.detrend(data_reg,axis=1,type='linear')
    elif detrend == 'RemoveMean':
        # - remove mean from the data
        data_reg = data_reg - data_reg.mean()
    elif detrend == 'RmeanDtrend':
        # - remove mean and detrend data in both direction
        data_reg = data_reg - data_reg.mean()
        data_reg = signal.detrend(data_reg,axis=0,type='linear')
        data_reg = signal.detrend(data_reg,axis=1,type='linear')
    
    # obtain dx and dy
    x1dreg,y1dreg = x_reg[0,:],y_reg[:,0]
    dx=np.int(np.ceil(x1dreg[1]-x1dreg[0]))
    dy=np.int(np.ceil(y1dreg[1]-y1dreg[0]))

    # wavenumber vector
    kspec,kh = wavenumber_vector(Ni,Nj,dx,dy)
    
    # Apply windowing
    if window is None:
        data_reg = data_reg
    elif window == 'han':
        wdw = hanning(Ni,Nj)
        data_reg*=wdw
    elif window == 'tukey':
        wdw = tukey(Ni,Nj)
        data_reg*=wdw
    elif window == 'scaled_han':
        wdw = scaled_han(Ni,Nj)
        data_reg*=wdw

    spec_2D = get_spec_2D(data_reg,dx,dy,Ni,Nj)
    spec_1D = get_spec_1D(kh,kspec,spec_2D)
    pspec = spec_1D
    return kspec,pspec

