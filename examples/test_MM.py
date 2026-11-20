import numpy as np
import gnlse
import gnlse.mm_gnlse as mm_gnlse
import matplotlib.pyplot as plt
import os
from gnlse.common import c
import copy

if __name__ == '__main__':
    setup = mm_gnlse.MMGNLSESetup()

    # Numerical parameters
    ###########################################################################
    # number of grid time points
    setup.resolution = 2**17
    # time window [ps]
    setup.time_window = 1600#12.5
    # number of distance points to save
    setup.z_saves = 200
    # relative tolerance for ode solver
    setup.rtol = 1e-6
    # absoulte tolerance for ode solver
    setup.atol = 1e-6
    setup.eps = 20

    # Physical parameters
    ###########################################################################
    # Central wavelength [nm]
    setup.wavelengths = [532, 653.432631684888, 448.627879567843]
    # Nonlinear coefficient [1/W/m]
    setup.gamma = {
        'gamma11': 0.0081,
        'gamma33': 0.0049,
        'gamma44': 0.0102,
        'gamma13': 0.0052,
        'gamma14': 0.0099,
        'gamma34': 0.0043,
        'gamma31': 0.0042,
        'gamma41': 0.0068,
        'gamma43': 0.0062,
        'gamma1134': 0.0042,
        'gamma3411': 0.0034,
        'gamma4311': 0.0050,
        'gammaFWM': 0.0041
    }

    # Dispersion: beta_01 and beta_02 are given in 1/m as well as the corresponding wavelengths in m. The dispersion operator is built from interpolation of the given data.
    dispersion_data_filename = os.path.join(os.path.dirname(__file__), '..',
                            'data', 'dispersion_data_SMF28.txt')
    
    lmbda, beta_01, beta_02 = np.loadtxt(dispersion_data_filename, delimiter='\t', unpack=True)
    n_effs = np.array([beta_01, beta_02, beta_01])*lmbda/2/np.pi
    # Input pulse: pulse duration [ps]
    tFWHM = 400

    ###########################################################################
    
    setup.num_channels = 3
    # Input pulse: peak power [W]
    power = 50e3  # value can be choosen arbitrarily
    # Non-linear length for given nonlinearity and power
    LNL = 1 / (power * setup.gamma['gamma11'])
    # Fiber length [m]
    setup.fiber_length = 0.5
    # Type of pulse:  gaussian
    setup.pulse_model = [None] * setup.num_channels
    setup.pulse_model[0] = gnlse.SechEnvelope(power, tFWHM)
    setup.pulse_model[1] = gnlse.CWEnvelope(0, Pn=1e-18) 
    setup.pulse_model[2] = gnlse.CWEnvelope(0, Pn=1e-18) 
    # Loss coefficient [dB/m]
    loss = 0
    # Type of dispersion operator: build from interpolation of given data
    setup.dispersion_models = gnlse.dispersion.TripleDispersionFiberFromInterpolation(
        loss, n_effs, lmbda*1e9, setup.wavelengths)

    # Type of Raman scattering function: None (default)
    # setup.raman_model = gnlse.raman_blowwood
    # Selfsteepening: not accounted
    setup.self_steepening = False

    # Simulation
    ###########################################################################
    solver = mm_gnlse.MMGNLSE(setup)
    solution = solver.run()

    # Visualization
    ###########################################################################
    WL = 2 * np.pi * c / solver.W
    plt.figure()
    print(WL.shape, solution.AW[-1, :, :].shape)
    plt.plot(WL.T, 10*np.log10(np.abs(solution.AW[0, :, :].T)**2), ':')
    plt.plot(WL.T, 10*np.log10(np.abs(solution.AW[-1, :, :].T)**2))
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Power (dB)')
    plt.title('Input and Output Spectrum')
    plt.xlim([400, 800])
    #plt.ylim([40, 100])
    plt.legend(('z = 0 m, channel 1', 'channel 2', 'channel 3', 'z = 0.5 m, channel 1', 'channel 2', 'channel 3'))
    plt.show()
    
    sol_ch1 = copy.deepcopy(solution)
    sol_ch1.W = solution.W[0, :]
    sol_ch1.AW = solution.AW[:, 0, :]
    sol_ch1.At = solution.At[:, 0, :]

    plt.subplot(1, 2, 1)
    gnlse.plot_wavelength_vs_distance_logarithmic(sol_ch1, WL_range=[400, 1400])
    plt.subplot(1, 2, 2)
    gnlse.plot_delay_vs_distance_logarithmic(sol_ch1, time_range=[-.25, .25])

    plt.show()

    sol_ch3 = copy.deepcopy(solution)
    sol_ch3.W = solution.W[1, :]
    sol_ch3.AW = solution.AW[:, 1, :]
    sol_ch3.At = solution.At[:, 1, :]

    plt.subplot(1, 2, 1)
    gnlse.plot_wavelength_vs_distance_logarithmic(sol_ch3, WL_range=[400, 1400])
    plt.subplot(1, 2, 2)
    gnlse.plot_delay_vs_distance_logarithmic(sol_ch3, time_range=[-.25, .25])

    plt.show()

    sol_ch4 = copy.deepcopy(solution)
    sol_ch4.W = solution.W[2, :]
    sol_ch4.AW = solution.AW[:, 2, :]
    sol_ch4.At = solution.At[:, 2, :]
    
    plt.subplot(1, 2, 1)
    gnlse.plot_wavelength_vs_distance_logarithmic(sol_ch4, WL_range=[400, 1400])
    plt.subplot(1, 2, 2)
    gnlse.plot_delay_vs_distance_logarithmic(sol_ch4, time_range=[-.25, .25])
    
    plt.show()

print('done')