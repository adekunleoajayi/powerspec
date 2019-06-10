## load modules
import numpy as np
import pandas as pd
import numpy.fft as fft
import numpy.ma as ma
import numpy.linalg as li
import scipy.signal as signal
from scipy.interpolate import interp1d

def _e1e2(navlon,navlat):
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

#########################
def interpolate(data,navlon,navlat,interp=None):
    """
        Perform a spatial interpolation if required; return x_reg,y_reg,data_reg.
        data : raw data
        nalon : longitude
        navlat : latitude
        interp : if None return data with cordinates in meters, if 'basemap', return interpolated
        data using basemap from mpl_toolkits and also cordinates in meters.
        """
    e1,e2 = _e1e2(navlon,navlat)
    x1d_in = e1[0,:].cumsum() - e1[0,0]
    y1d_in = e2[:,0].cumsum() - e2[0,0]
    x2d_in,y2d_in = np.meshgrid(x1d_in,y1d_in)
    # print x1d_in
    if interp is None :
        return x2d_in, y2d_in, data
    elif interp=='basemap': # only for rectangular grid...
        from mpl_toolkits import basemap
        x1d_reg=np.linspace(x1d_in[0],x1d_in[-1],len(x1d_in))
        y1d_reg=np.linspace(y1d_in[0],y1d_in[-1],len(y1d_in))
        x2d_reg,y2d_reg = np.meshgrid(x1d_reg,y1d_reg)
        data_reg=basemap.interp(data,x1d_in,y1d_in,x2d_reg,y2d_reg,checkbounds=False,order=1)
        return x2d_reg,y2d_reg,data_reg
    else: raise ValueError('Your choice of interp is not available in this sript.')

#########################
def isdata_contain_nan(data):
    '''
        This function check if a data contains any NaN value
        If yes, it replaces the NaN values with an interpolated
        value using the fill_nan function.
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

#########################
def fill_nan(data):
    ''' replaces a NaN value in a dataset with an interpolated one.
        
        return :
        data_nonan
        '''
    # - Reshape to 1D array
    i,j = data.shape
    _1D_arr = data.reshape(i*j,1)
    _1D_arr = _1D_arr.squeeze()
    # - get nan
    nt_nan = np.logical_not(np.isnan(_1D_arr))
    # - get array length
    indice = np.arange(len(_1D_arr))
    # - interpolate for nan value only
    interp = interp1d(indice[nt_nan], _1D_arr[nt_nan],kind ='linear',fill_value="extrapolate")
    # - replace nan values
    arr = interp(indice)
    # - reshape back to 2D array
    data_nonan = arr.reshape(i,j)
    return data_nonan

#########################
def _tukey(Ni,Nj):
    ''' Using tukey window : tapered cosine window. /alpha = 0.5'''
    wdwi =  signal.tukey(Ni,0.5,sym=False)
    wdwj =  signal.tukey(Nj,0.5,sym=False)
    wdw = wdwi[np.newaxis,...]*wdwj[...,np.newaxis]
    return wdw

#########################
def _hanning(Ni,Nj):
    ''' Using Hanning window'''
    wdwi =  signal.hanning(Ni,sym=False)
    wdwj =  signal.hanning(Nj,sym=False)
    wdw = wdwi[np.newaxis,...]*wdwj[...,np.newaxis]
    return wdw

#########################
def _wavenumber_vector(x,y,Ni,Nj):
    ''' Compute a wavenumber vector  '''
    # - obtain dx and dy
    x1dreg,y1dreg = x[0,:],y[:,0]
    dx=np.int(np.ceil(x1dreg[1]-x1dreg[0]))
    dy=np.int(np.ceil(y1dreg[1]-y1dreg[0]))
    
    kx = np.fft.fftfreq(Ni,dx) # two sided
    ky = np.fft.fftfreq(Nj,dy)
    Kmax = max(kx.max(), ky.max())
    
    k, l = np.meshgrid(kx,ky)
    wavnum2D = np.sqrt(k**2 + l**2)
    
    # radial wavenumber
    ddk = 1./(dx*Ni)
    ddl = 1./(dy*Nj)
    
    dK = max(ddk,ddl)
    wavnum1D = dK*np.arange(1,int(Kmax/dK))
    return wavnum1D,wavnum2D,kx,ky,dx,dy

#########################
def _get_2D_psd(data_reg,dx,dy,Ni,Nj):
    ''' Compute the 2D spectrum of the data '''
    spec_fft = np.fft.fft2(data_reg)
    spec_2D = (spec_fft*spec_fft.conj()).real*(dx*dy)/(Ni*Nj)
    return spec_2D

#########################
def _get_1D_psd(kradial,wavnum,spec_2D):
    ''' Compute the azimuthaly avearge of the 2D spectrum '''
    spec_1D = np.zeros(len(wavnum))
    for i in range(wavnum.size):
        kfilt =  (kradial>=wavnum[i] - wavnum[0]) & (kradial<=wavnum[i])
        N = kfilt.sum()
        spec_1D[i] = (spec_2D[kfilt].sum())*wavnum[i]/N
    return spec_1D

#########################
def _detrend_data(x,y,data,detrend):
    if detrend is None :
        data = data
        return data
    elif detrend == 'Both':
        # - detrend data in both direction
        data = signal.detrend(data,axis=0,type='linear')
        data = signal.detrend(data,axis=1,type='linear')
        return data
    elif detrend == 'Zonal':
        # - detrend data in the zonal direction
        data = signal.detrend(data,axis=1,type='linear')
        return data
    elif detrend == 'RemoveMean':
        # - remove mean from the data
        data = data - data.mean()
        return data
    elif detrend == 'RmeanDtrend':
        # - remove mean and detrend data in both direction
        data = data - data.mean()
        data = signal.detrend(data,axis=0,type='linear')
        data = signal.detrend(data,axis=1,type='linear')
        return data
    else: raise ValueError('Your choice of detrend is not available in this sript.')

#########################
def _apply_windowing(data,window):
    Nj,Ni = data.shape
    if window is None:
        data = data
        return data
    elif window == 'Hanning':
        wdw = _hanning(Ni,Nj)
        data*=wdw
        return data
    elif window == 'Tukey':
        wdw = _tukey(Ni,Nj)
        data*=wdw
        return data
    else: raise ValueError('Your choice of windowing is not available in this sript.')

#########################
def preprocess_data(data,detrend,window,navlon,navlat):
    x,y,data = interpolate(data,navlon,navlat)  # get x and y cordinates
    data = isdata_contain_nan(data)             # check if data contains NaN value
    data = _detrend_data(x,y,data,detrend)      # Detrend
    data = _apply_windowing(data,window)        # Apply windowing
    return data,x,y

#########################
def _get_transfer1D(wavnum2D,wavnum1D,spec_2D):
    ''' Compute KE Transfer'''
    transfer = np.zeros(len(wavnum1D))
    for i in range(wavnum1D.size):
        kfilt =  ((wavnum1D[i] - wavnum1D[0]) <= wavnum2D) & (wavnum2D <= wavnum1D[i])
        transfer[i] = (spec_2D[kfilt]).sum()
    return transfer

#########################
def _get_grad_uv_in_fourier_space(u,v,kx,ky,Ni,Nj):
    ''' gradient of u and v in spectral space
        return dudx,dudy,dvdx,dvdy
        '''
    dudx = np.real(np.fft.ifft2(np.fft.fft2(u)*(1j*kx*2*np.pi)[None, :]));
    dudy = np.real(np.fft.ifft2(np.fft.fft2(u)*(1j*ky*2*np.pi)[:, None]));
    dvdx = np.real(np.fft.ifft2(np.fft.fft2(v)*(1j*kx*2*np.pi)[None, :]));
    dvdy = np.real(np.fft.ifft2(np.fft.fft2(v)*(1j*ky*2*np.pi)[:, None]));
    return dudx,dudy,dvdx,dvdy

#########################
def wavenumber_spectra(data,navlon,navlat,window='Tukey',detrend='Both'):
    """ This function computes wavenumber spectral density of a two dimensional dataset.
        
        Input
            data : Two dimensional dataset Eg SSH
            navlon : Longitude
            navlat : Latitude
            window : None , 'Hanning' or 'Tukey' : (tappered consine window with /apha = 0.5).
            detrend :
                "both" : detrend the 2D data along both axes.
                "zonal" : detrend the data in the zonal direction only
                "RemoveMean" : Remove only the mean of the data
                "RmeanDtrend" : Remove the mean then detrend the data in both direction
                None : use the raw data
        Return
            wavenumber : Horizontal wavenumber
            psd : isotropic power spectral density
        """
    Nj,Ni = data.shape                                                      # Get data shape
    data,x,y = preprocess_data(data,detrend,window,navlon,navlat)           # Remove NaN, Detrend, apply windowing
    wavenumber,wavenumber_2D,kx,ky,dx,dy = _wavenumber_vector(x,y,Ni,Nj)    # Get wavenumber
    spec_2D = _get_2D_psd(data,dx,dy,Ni,Nj)                                 # Estimate 2D spectra
    psd = _get_1D_psd(wavenumber_2D,wavenumber,spec_2D) # Average 2D spectra to 1D spectra : isotropic wavenumber spectra density
    return wavenumber,psd


#########################
def spectra_flux(_u,_v,navlon,navlat,window='Tukey',detrend='Both'):
    '''This fuctions implement the computation of kinetic energy spectral flux.
        flux = \int^{ks}_{k} u ....
        
        Input
            _u : Two dimensional zonal velocity dataset.
            _v : Two dimensional meriodinal velocity dataset.
            navlon : Longitude
            navlat : Latitude
            window : None , 'Hanning' or 'Tukey' : (tappered consine window with /apha = 0.5).
            detrend :
                "both" : detrend the 2D data along both axes.
                "zonal" : detrend the data in the zonal direction only
                "RemoveMean" : Remove only the mean of the data
                "RmeanDtrend" : Remove the mean then detrend the data in both direction
                None : use the raw data
        Return
            wavenumber : Horizontal wavenumber
            flux : kinetic energy spectral flux
        '''
    Nj,Ni = _u.shape                                             # Get data shape
    u,x,y = preprocess_data(_u,detrend,window,navlon,navlat)     # Remove NaN, Detrend, apply windowing
    v,x,y = preprocess_data(_v,detrend,window,navlon,navlat)     # Remove NaN, Detrend, apply windowing
    wavenumber,wavenumber_2D,kx,ky,dx,dy = _wavenumber_vector(x,y,Ni,Nj)
    dudx,dudy,dvdx,dvdy = _get_grad_uv_in_fourier_space(u,v,kx,ky,Ni,Nj) # Get gradient of u and v
    # - compute terms
    phi1 = u*dudx + v*dudy
    phi2 = u*dvdx + v*dvdy
    # - compute FFT
    tm1 = (fft.fft2(u).conj())*fft.fft2(phi1)
    tm2 = (fft.fft2(v).conj())*fft.fft2(phi2)
    
    transfer_2D = -1*np.real(tm1 + tm2)/np.square(Ni*Nj) # - KE transfer
    transfer_1D = _get_transfer1D(wavenumber_2D,wavenumber,transfer_2D) # Get 1D transfer : assume isotropy
    flux = np.cumsum(transfer_1D[::-1])[::-1] # - Get flux
    return wavenumber,flux


def cross_spectra(data1,data2,navlon,navlat,window='Tukey',detrend='Both',Normalize=True):
    '''This fuctions compute the spectra coherence between two 2D datasets
        
        Input
            _u : Two dimensional zonal velocity dataset.
            _v : Two dimensional meriodinal velocity dataset.
            navlon : Longitude
            navlat : Latitude
            window : None , 'Hanning' or 'Tukey' : (tappered consine window with /apha = 0.5).
            detrend :
                "both" : detrend the 2D data along both axes.
                "zonal" : detrend the data in the zonal direction only
                "RemoveMean" : Remove only the mean of the data
                "RmeanDtrend" : Remove the mean then detrend the data in both direction
                None : use the raw data
        Return
            wavenumber : Horizontal wavenumber
            cross spectra : spectra coherence between data1 and data2
        '''
    Nj,Ni = data1.shape                                                 # Get data shape
    data1,x,y = preprocess_data(data1,detrend,window,navlon,navlat)     # Remove NaN, Detrend, apply windowing
    data2,x,y = preprocess_data(data2,detrend,window,navlon,navlat)     # Remove NaN, Detrend, apply windowing
    wavenumber,wavenumber_2D,kx,ky,dx,dy = _wavenumber_vector(x,y,Ni,Nj)        # wavenumber vector
    spec_2D = (np.fft.fft2(data1) * np.fft.fft2(data2).conj()).real*(dx*dy)/(Ni*Nj) # Cross spectra
    
    cross_spectra = _get_1D_psd(wavenumber_2D,wavenumber,spec_2D) # Average 2D spectra to 1D spectra
    
    if Normalize:
        cs1_2D = _get_2D_psd(data1,dx,dy,Ni,Nj); cs1_1D = _get_1D_psd(wavenumber_2D,wavenumber,cs1_2D)
        cs2_2D = _get_2D_psd(data2,dx,dy,Ni,Nj); cs2_1D = _get_1D_psd(wavenumber_2D,wavenumber,cs2_2D)
        cross_spectra = (abs(cross_spectra)**2)/(cs1_1D*cs2_1D)
    return wavenumber,cross_spectra


def weighted_scale(wavenumber,psd,axs=None):
    '''Compute weighted scale from wavenumber spectra density.
        just like integral scale
        input :
            wavenumber ==> wavenumber
            psd ==> Power spectral density
        return :
            scale ==> weighted scale
        '''
    if len(psd.shape) == 2:
        if not axs: raise ValueError('Provide axis to sum over')
        return psd.sum(axis=axs)/(wavenumber*psd).sum(axis=axs)
    if len(psd.shape) > 2:
        raise ValueError('Dimension of psd is more than 2')
    else:
        return psd.sum()/(wavenumber*psd).sum()
