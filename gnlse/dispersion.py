"""Dispersion operator in optical fibres.

Based on delivered frequency vector and damping indexes
script calculates linear dispersion operator in
frequency domain.

"""
import numpy as np
import math
from scipy import interpolate

from gnlse.common import c


class Dispersion(object):
    """
    Attributes
    -----------
    loss : float
        Loss factor [dB/m]
    """

    def __init__(self, loss):
        self.loss = loss

    def D(V):
        """Calculate linear dispersion operator
        for given frequency grid created during simulation

        Parameters
        ----------
        V : ndarray, (N)
            Frequency vector

        Returns
        -------
        ndarray, (N)
            Linear dispersion operator in frequency domain
        """

        raise NotImplementedError('Dispersion not implemented')

    def calc_loss(self):
        """Calculate damping
        for given frequency grid created during simulation
        """
        self.alpha = np.log(10**(self.loss / 10))


class DispersionFiberFromTaylor(Dispersion):
    """Calculates the dispersion in frequency domain

    Attributes
     ----------
    loss : float
        Loss factor [dB/m]
    betas : ndarray (N)
        Derivatives of constant propagations at pump wavelength
        [ps^2/m, ..., ps^n/m]
    """

    def __init__(self, loss, betas):
        self.loss = loss
        self.betas = betas

    def D(self, V):
        # Damping
        self.calc_loss()
        # Taylor series for subsequent derivatives
        # of constant propagation
        B = sum(beta / math.factorial(i + 2) * V**(i + 2)
                for i, beta in enumerate(self.betas))
        L = 1j * B - self.alpha / 2
        return L


class DispersionFiberFromInterpolation(Dispersion):
    """Calculates the propagation function in frequency domain, using
    the extrapolation method based on delivered refractive indexes
    and corresponding wavelengths. The returned value is a vector
    of dispersion operator.

    Attributes
    -----------
    loss : float
        Loss factor [dB/m]
    neff : ndarray (N)
        Effective refractive index
    lambdas : ndarray (N)
        Wavelength corresponding to refractive index
    central_wavelength : float
        Wavelength corresponding to pump wavelength in nm
    """

    def __init__(self, loss, neff, lambdas, central_wavelength):
        # Loss factor in dB/m
        self.loss = loss
        # refractive indices
        self.neff = neff
        # wavelengths for neffs in [nm]
        self.lambdas = lambdas
        # Central frequency in [1/ps = THz]
        self.w0 = (2.0 * np.pi * c) / central_wavelength

    def D(self, V):
        # Central frequency [1/ps = THz]
        omega = 2 * np.pi * c / self.lambdas
        dOmega = V[1] - V[0]
        Bet = self.neff * omega / c * 1e9  # [1/m]

        # Extrapolate betas for a frequency vector
        fun_interpolation = interpolate.interp1d(omega,
                                                 Bet,
                                                 kind='cubic',
                                                 fill_value="extrapolate")

        B = fun_interpolation(V + self.w0)
        # Propagation constant at central frequency [1/m]
        B0 = fun_interpolation(self.w0)
        # Value of propagation at a lower end of interval [1/m]
        B0plus = fun_interpolation(self.w0 + dOmega)
        # Value of propagation at a higher end of interval [1/m]
        B0minus = fun_interpolation(self.w0 - dOmega)

        # Difference quotient, approximation of
        # derivative of a function at a point [ps/m]
        B1 = (B0plus - B0minus) / (2 * dOmega)

        # Damping
        self.calc_loss()

        # Linear dispersion operator
        L = 1j * (B - (B0 + B1 * V)) - self.alpha / 2
        return L

class TripleDispersionFiberFromInterpolation(Dispersion):
    """Calculates the propagation function in frequency domain, using
    the extrapolation method based on delivered refractive indexes
    and corresponding wavelengths. The returned value is a vector
    of dispersion operator.

    Attributes
    -----------
    loss : float
        Loss factor [dB/m]
    neffs : ndarray (N)
        Effective refractive indices 
    lambdas : ndarray (N)
        Wavelength corresponding to refractive index
    central_wavelength : float
        Wavelength corresponding to pump wavelength in nm
    """

    def __init__(self, loss, neffs, lambdas, central_wavelengths):
        # Loss factor in dB/m
        self.loss = loss
        # refractive indices
        self.neff1 = neffs[0, :]
        self.neff3 = neffs[1, :]
        self.neff4 = neffs[2, :]
        # wavelengths for neffs in [nm]
        self.lambdas = lambdas
        # Central frequency in [1/ps = THz]
        self.w1 = (2.0 * np.pi * c) / central_wavelengths[0]
        self.w3 = (2.0 * np.pi * c) / central_wavelengths[1]
        self.w4 = (2.0 * np.pi * c) / central_wavelengths[2]

    def D(self, V):
        # Central frequency [1/ps = THz]
        omega = 2 * np.pi * c / self.lambdas
        dOmega = V[1] - V[0]
        Bet_m1 = self.neff1 * omega / c * 1e9  # [1/m]
        Bet_m3 = self.neff3 * omega / c * 1e9  # [1/m]
        Bet_m4 = self.neff4 * omega / c * 1e9  # [1/m]

        # Extrapolate betas for a frequency vector
        fun_interpolation_m1 = interpolate.interp1d(omega,
                                                 Bet_m1,
                                                 kind='cubic',
                                                 fill_value="extrapolate")
        fun_interpolation_m3 = interpolate.interp1d(omega,
                                                 Bet_m3,
                                                 kind='cubic',
                                                 fill_value="extrapolate")
        fun_interpolation_m4 = interpolate.interp1d(omega,
                                                 Bet_m4,
                                                 kind='cubic',
                                                 fill_value="extrapolate")

        B_m1 = fun_interpolation_m1(V + self.w1)
        B_m3 = fun_interpolation_m3(V + self.w3)
        B_m4 = fun_interpolation_m4(V + self.w4)
        # Propagation constant at central frequency [1/m]
        B0_m1 = fun_interpolation_m1(self.w1)
        B0_m3 = fun_interpolation_m3(self.w3)
        B0_m4 = fun_interpolation_m4(self.w4)
        # Value of propagation at a lower end of interval [1/m]
        B0plus = fun_interpolation_m1(self.w1 + dOmega)
        # Value of propagation at a higher end of interval [1/m]
        B0minus = fun_interpolation_m1(self.w1 - dOmega)

        # Difference quotient, approximation of
        # derivative of a function at a point [ps/m]
        B1 = (B0plus - B0minus) / (2 * dOmega)

        # Damping
        self.calc_loss()

        # Linear dispersion operator
        L1 = 1j * (B_m1 - (B0_m1 + B1 * V)) - self.alpha / 2
        L3 = 1j * (B_m3 - (B0_m3 + B1 * V)) - self.alpha / 2
        L4 = 1j * (B_m4 - (B0_m4 + B1 * V)) - self.alpha / 2

        # phase shift in as a function of frequency
        deltabeta = (
            -2*B0_m1 + fun_interpolation_m3(self.w3 - V) 
                     + fun_interpolation_m4(self.w4 + V)
                     )

        return [L1, L3, L4, deltabeta]

class TripleDispersionFiberFromTaylor(Dispersion):
    """Calculates the dispersion in frequency domain

    Attributes
     ----------
    loss : float
        Loss factor [dB/m]
    betas : ndarray (N)
        Derivatives of constant propagations at pump wavelength
        [ps^2/m, ..., ps^n/m]
    """
    # not made for more than 1 mode at the moment, just made to work with MM_GNLSE code

    def __init__(self, loss, betas):
        self.loss = loss
        self.betas = betas

    def D(self, V):
        # Damping
        self.calc_loss()
        # Taylor series for subsequent derivatives
        # of constant propagation
        B = sum(beta / math.factorial(i + 2) * V**(i + 2)
                for i, beta in enumerate(self.betas))
        L1 = 1j * B - self.alpha / 2   

        if self.deltabeta is None:
            deltabeta = 0
        else:
            deltabeta = self.deltabeta
        return [L1, L1, L1, deltabeta]
    