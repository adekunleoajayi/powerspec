
This scripts estimate wavenumber spectrum of two dimensional oceanic dataset such as SSH, vorticity. Prior to spectrum estimate, the dataset is interpolated and any NaN value is replaced with interpolated value. A tapering is applied to the dataset, afterwhich, the dataset is detrend in both direction. The 2D spectral obtained after FFT is radially averaged to a 1D spectral.

Check the "AJ_2017_09_22_Example_on_how_to_use_PowerSpec" notebook for useage.


############################################################################</br>
Author : Adekunle Ajayi and Julien Lesommer</br>
Affilation : Institut des Geosciences de l'Environnement (IGE), Universite Grenoble Alpes, France.</br>
Email : adekunle.ajayi@univ-grenoble-alpes.fr, julien.lesommer@univ-grenoble-alpes.fr </br>
############################################################################ </br>
