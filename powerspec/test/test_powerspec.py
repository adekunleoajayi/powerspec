import unittest
import sys
import xarray as xr
import numpy as np

data = xr.open_dataset('./test_data.nc')
ssh = data.ssh
u = data.u
v = data.v
u15 = data.u15

sys.path.insert(0,'../')
import powerspec as ps


class TestPowerSpec(unittest.TestCase):

    def test_spectral_density(self):
        wavenumber, psd = ps.wavenumber_spectra(ssh.to_masked_array(), ssh.nav_lon.data, ssh.nav_lat.data)
        self.assertEqual(len(wavenumber), 298, "Should be 298")
        self.assertEqual(round(psd.max(),2),241.94,"Should be 241.94")

    def test_spectra_flux(self):
        wavenumber, flux = ps.spectra_flux(u.to_masked_array(),v.to_masked_array(), u.nav_lon.data, u.nav_lat.data)
        self.assertEqual(len(wavenumber), 298, "Should be 298")
        self.assertEqual(round(np.log10(flux.mean()),2),-14.67,"Should be -14.67")

    def test_cross_spectra(self):
        wavenumber, cs = ps.cross_spectra(u.to_masked_array(),u15.to_masked_array(), u.nav_lon.data, u.nav_lat.data,Normalize=True)
        self.assertEqual(len(wavenumber), 298, "Should be 298")
        self.assertEqual(round(cs.mean(),4),0.0155,"Should be 0.0155")
                      

if __name__ == '__main__':
    unittest.main()
