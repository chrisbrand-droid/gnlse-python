import numpy as np
import scipy.integrate
import pyfftw
import tqdm

from gnlse.common import c
from gnlse.import_export import write_mat, read_mat

import matplotlib.pyplot as plt

class MMGNLSESetup:
    """
    Model inputs for the ``MMGNLSE`` class.

    Attributes
    ----------
    resolution : int
        Number of points on the computational grid. Determines time resolution
        and bandwidth. Avoid numbers with large prime factors.
    time_window : float [ps]
        Width of the time window.
    wavelength_s : float [nm]
        Central wavelengths of the input pulse and stokes and antistokes peaks.
    fiber_length : float [m]
        Length of the simulated optical fiber.
    z_saves : int
        Number of snapshots to save along the fiber. Larger numbers require
        more memory to store the result.
    nonlinearity : float [1/W/m]
        Effective nonlinearity.
    pulse_model : Envelope
        Input pulse envelope model.
    dispersion_model : Dispersion, optional
        Fiber dispersion model or ``None`` to model a dispersionless fiber.
    raman_model : function, optional
        Raman scattering model or ``None`` if the effect is to be neglected.
    self_steepning : bool, optional
        Whether to include the effect of self-steepening. Disabled by default.
    rtol : float, optional
        Relative tolerance passed to the ODE solver.
    atol : float, optional
        Absolute tolerance passed to the ODE solver.
    method : str, optional
        Integration method passed to the ODE solver.
    """

    def __init__(self):
        # Number of frequency channels
        self.num_channels = 3

        self.resolution = None
        self.time_window = None
        self.wavelengths = None
        self.fiber_length = None

        self.z_saves = 200
        self.gamma = None
        self.deltabeta = None
        self.pulse_model = None
        self.dispersion_models = None
        self.raman_model = None
        self.self_steepening = False

        self.rtol = 1e-3
        self.atol = 1e-4
        self.method = 'RK45'
        self.eps = 1


class Solution:
    """
    Represents a solution to a GNLSE problem.

    Attributes
    ----------
    t : ndarray, (n,)
        Time domain grid.
    W_p : ndarray, (n,)
        Absolute angular frequency grid.
    Z : ndarray (m,)
        Points at which intermediate steps were saved.
    At : ndarray, (n, m)
        Intermediate steps in the time domain.
    AW : ndarray, (n, m)
        Intermediate steps in the frequency domain.
    """

    def __init__(self, t=None, W=None, w_0=None,
                 Z=None, At=None, AW=None):
        self.t = t
        self.W = W
        self.w_0 = w_0
        self.Z = Z
        self.At = At
        self.AW = AW

    def to_file(self, path):
        """
        Saves a solution to a file.

        Parameters
        ----------
        path : str
            Path to file.
        """

        data = {'t': self.t, 'W': self.W, 
                'w_0': self.w_0, 'Z': self.Z, 
                'At': self.At, 'AW': self.AW}
        write_mat(data, path)

    def from_file(self, path):
        """
        Load a solution from file.

        Parameters
        ----------
        path : str
            Path to file.
        """

        data = read_mat(path)
        self.t = data['t']
        self.W = data['W']
        self.w_0 = data['w_0']
        self.Z = data['Z']
        self.At = data['At']
        self.AW = data['AW']


class MMGNLSE:
    """
    Models propagation of an optical pulse in a fiber by integrating
    the generalized non-linear Schrödinger equation.

    Attributes
    ----------
    setup : MMGNLSESetup
        Model inputs in the form of a ``MMGNLSESetup`` object.
    """

    def __init__(self, setup):
        if not isinstance(setup, MMGNLSESetup):
            raise TypeError("setup is not an instance of MMGNLSESetup")

        self.setup = setup

        if setup.resolution is None:
            raise ValueError("'resolution' not set")
        if setup.time_window is None:
            raise ValueError("'time_window' not set")
        if setup.wavelengths is None:
            raise ValueError("'wavelengths' not set")
        if setup.fiber_length is None:
            raise ValueError("'fiber_length' not set")
        if setup.pulse_model is None:
            raise ValueError("'pulse_model' not set")
        if setup.dispersion_models is None:
            raise ValueError("'dispersion_models' not set")
        if setup.gamma is None:
            raise ValueError("'gamma' not set")

        # simulation parameters
        self.fiber_length = setup.fiber_length
        self.z_saves = setup.z_saves
        self.rtol = setup.rtol
        self.atol = setup.atol
        self.method = setup.method
        self.N = setup.resolution
        self.n_modes = setup.num_channels

        # Time domain grid (common to all channels)
        self.t = np.linspace(-setup.time_window / 2,
                             setup.time_window / 2,
                             self.N)

        # Relative angular frequency grid
        self.V = 2 * np.pi * np.fft.fftfreq(
            self.N, d=(self.t[1] - self.t[0])
            )
        # Central angular frequencies [10^12 rad]
        self.w_0 = (2.0 * np.pi * c) / np.asarray(setup.wavelengths)
        # Absolute angular frequency grid for each channel
        self.W = self.V[None, :] + self.w_0[:, None]

        # Absolute angular frequency grid
        if setup.self_steepening:
            self.shock = self.W/self.w_0[:, None]
        else:
            self.shock = np.ones_like(self.W)

        self.gamma = setup.gamma
        self.eps = setup.eps

        self.scale = 1.0  # scale factor for the nonlinearity, can be adjusted if needed
        # Nonlinearity
        #if hasattr(setup.nonlinearity, 'gamma'):
        #    # in case in of frequency dependent nonlinearity
        #    gamma, self.scale = setup.nonlinearity.gamma(self.V)
        #    self.gamma = gamma / self.w_p
        #    self.gamma = np.fft.fftshift(self.gamma)
        #    self.scale = np.fft.fftshift(self.scale)
        #else:
        #    # in case in of direct introduced value
        #    self.gamma = setup.nonlinearity / self.w_p
        #    self.scale = 1

        # Raman scattering
        self.RW = None
        if setup.raman_model:
            self.fr, RT = setup.raman_model(self.t)
            if np.abs(self.fr) < np.finfo(float).eps:
                self.RW = None
            else:
                self.RW = self.N * np.fft.ifft(
                    np.fft.fftshift(np.transpose(RT)))

        # Dispersion operator
        self.D = np.zeros((self.n_modes, self.N),
                          dtype=np.complex128)

        if setup.dispersion_models is not None:
            L1, L3, L4, deltabeta = setup.dispersion_models.D(np.fft.fftshift(self.V))
        self.D[0, :] = np.fft.fftshift(L1)
        self.D[1, :] = np.fft.fftshift(L3)
        self.D[2, :] = np.fft.fftshift(L4)
        self.deltabeta = np.fft.fftshift(deltabeta)
        self.deltabeta[0] = 0

        # Input pulse
        if hasattr(setup.pulse_model[0], 'A'):
            if self.n_modes == 1:
                self.A = setup.pulse_model.A(self.t)
            else:
                self.A = np.zeros((self.n_modes, self.N), dtype=np.complex128)
                for j in range(self.n_modes):
                    self.A[j, :] = setup.pulse_model[j].A(self.t)
        else:
            self.A = np.zeros((self.n_modes, self.N), dtype=np.complex128)
            self.A[0, :] = setup.pulse_model

    def run(self):
        """
        Solve multimode mode GNLSE equation (for degenerate IMFWM) described by the given
        ``MMGNLSESetup`` object.

        Returns
        -------
        setup : Solution
            Simulation results in the form of a ``Solution`` object.
        """
        dt = self.t[1] - self.t[0]
        #self.D = np.fft.fftshift(self.D) # V was already fftshifted when created, no need to shift again
        x = pyfftw.empty_aligned((self.n_modes, self.N), dtype="complex128")
        X = pyfftw.empty_aligned((self.n_modes, self.N), dtype="complex128")
        plan_forward = pyfftw.FFTW(x, X)
        plan_inverse = pyfftw.FFTW(X, x, direction="FFTW_BACKWARD")

        progress_bar = tqdm.tqdm(total=round(self.fiber_length, 3), unit='m')

        def rhs(z, AW):
            """
            The right hand side of the differential equation to integrate.
            """
            AW = AW.reshape((self.n_modes, self.N))

            progress_bar.n = round(z, 3)
            progress_bar.update(0)

            x[:, :] = AW * np.exp(self.D * z)
            At = plan_forward().copy()
            IT = np.abs(At)**2

            phase_1 = np.zeros_like(self.deltabeta, dtype = "complex128")
            phase_3 = np.zeros_like(self.deltabeta, dtype = "complex128")
            phase_4 = np.zeros_like(self.deltabeta, dtype = "complex128")

            if np.abs(self.deltabeta.any()) < self.eps:
                ii = np.abs(self.deltabeta)<self.eps
                phase_1[ii] = np.exp( 1j * self.deltabeta[ii] * z)
                phase_3[ii] = np.exp( - 1j * self.deltabeta[ii] * z)
                phase_4[ii] = phase_3[ii]
                
                


            if self.RW is not None:
                X[0, :] = self.gamma['gamma11']*IT[0, :] + 2*self.gamma['gamma13']*IT[1, :] + 2*self.gamma['gamma14']*IT[2, :]
                X[1, :] = self.gamma['gamma33']*IT[1, :] + 2*self.gamma['gamma31']*IT[0, :] + 2*self.gamma['gamma34']*IT[2, :]
                X[2, :] = self.gamma['gamma44']*IT[2, :] + 2*self.gamma['gamma41']*IT[0, :] + 2*self.gamma['gamma43']*IT[1, :]
                plan_inverse()
                x[:, :] *= self.RW
                plan_forward()
                SPM_XPM = At*X

                X[0, :] = self.gamma['gamma1134']*At[1, :]*np.conj(At[0, :])
                X[1, :] = self.gamma['gamma3411']*At[0, :]*np.conj(At[2, :])
                X[2, :] = self.gamma['gamma4311']*At[0, :]*np.conj(At[1, :])
                plan_inverse()
                x[:, :] *= self.RW
                plan_forward()
                FWM = np.zeros((self.n_modes, self.N), dtype=np.complex128)
                FWM[0, :] = At[2, :]*X[0, :]*phase_1
                FWM[1, :] = 2*At[0, :]*X[1, :]*phase_3
                FWM[2, :] = 2*At[0, :]*X[2, :]*phase_4

                X[0, :] = self.gamma['gamma1134']*At[2, :]*np.conj(At[0, :])
                plan_inverse()
                x[0, :] *= self.RW
                plan_forward()
                FWM[0, :] += At[1, :]*X[0, :]*phase_1

                X[:, :] = (1-self.fr)*SPM_XPM + self.fr*(FWM)
                M = plan_inverse()

                RS1 = dt * self.fr * X[0, :]
                RS3 = dt * self.fr * X[1, :]
                RS4 = dt * self.fr * X[2, :]
                X1 = (
                    (self.gamma['gamma11']*IT[0, :] + 
                    2*self.gamma['gamma13']*IT[1, :] + 
                    2*self.gamma['gamma14']*IT[2, :])*At[0, :] + 
                    2*self.gamma['gamma1134']*At[1, :]*At[2, :]*np.conj(At[0, :])*phase_1)
                X3 = (
                    (self.gamma['gamma33']*IT[1, :] + 
                    2*self.gamma['gamma31']*IT[0, :] + 
                    2*self.gamma['gamma34']*IT[2, :])*At[1, :] + 
                    2*self.gamma['gamma3411']*At[0, :]**2*np.conj(At[2, :])*phase_3)
                X4 = (
                    (self.gamma['gamma44']*IT[2, :] + 
                    2*self.gamma['gamma41']*IT[0, :] + 
                    2*self.gamma['gamma43']*IT[1, :])*At[2, :] + 
                    2*self.gamma['gamma4311']*At[0, :]**2*np.conj(At[1, :])*phase_4)
                X[:, :] = (1-self.fr)*np.array([X1, X3, X4]) + np.array([RS1, RS3, RS4])
                M = plan_inverse()
            else:
                X[0, :] = (
                    (self.gamma['gamma11']*IT[0, :] + 
                    2*self.gamma['gamma13']*IT[1, :] + 
                    2*self.gamma['gamma14']*IT[2, :])*At[0, :] + 
                    2*self.gamma['gamma1134']*At[1, :]*At[2, :]*np.conj(At[0, :]))*phase_1
                X[1, :] = (
                    (self.gamma['gamma33']*IT[1, :] + 
                    2*self.gamma['gamma31']*IT[0, :] + 
                    2*self.gamma['gamma34']*IT[2, :])*At[1, :] + 
                    2*self.gamma['gamma3411']*At[0, :]**2*np.conj(At[2, :]))*phase_3
                X[2, :] = (
                    (self.gamma['gamma44']*IT[2, :] + 
                    2*self.gamma['gamma41']*IT[0, :] + 
                    2*self.gamma['gamma43']*IT[1, :])*At[2, :] + 
                    2*self.gamma['gamma4311']*At[0, :]**2*np.conj(At[1, :]))*phase_4
                M = plan_inverse()

            rv = 1j * self.shock * M * np.exp(
                -self.D * z)

            return rv.flatten()

        Z = np.linspace(0, self.fiber_length, self.z_saves)
        solution = scipy.integrate.solve_ivp(
            rhs,
            t_span=(0, self.fiber_length),
            y0 = np.fft.ifft(self.A, axis=1).flatten() * self.scale,
            t_eval=Z,
            rtol=self.rtol,
            atol=self.atol,
            method=self.method)
        AW = solution.y.T.reshape(
            self.z_saves, self.n_modes, self.N
            )

        progress_bar.close()

        # Transform the interaction-picture solution into the physical spectrum
        AW *= np.exp(self.D[None, :, :] * Z[:, None, None]) 
        AW /= self.scale

        # Transform each channel independently into the time domain
        At = np.fft.fft(AW, axis=2)

        # Shift the frequency axis for convenient plotting
        AW = np.fft.fftshift(AW, axes=2) * self.N * dt
        self.W = np.fft.fftshift(self.W, axes=1)

        return Solution(self.t, self.W, self.w_0, Z, At, AW)
