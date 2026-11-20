"""
Example: Evolution of the spectral and temporal characteristics of the
higher-order N = 3 soliton in three cases:
    - propagation without self steppening and Raman response;
    - soliton fission with self steppening, but no Raman response accounted;
    - soliton fission with self steppening, and Raman response accounted.
"""

import numpy as np
import matplotlib.pyplot as plt

import gnlse
import gnlse.mm_gnlse as mm_gnlse

if __name__ == '__main__':
    setup = mm_gnlse.MMGNLSESetup()

    # Numerical parameters
    # number of grid time points
    setup.resolution = 2**13
    # time window [ps]
    setup.time_window = 12.5
    # number of distance points to save
    setup.z_saves = 200
    # relative tolerance for ode solver
    setup.rtol = 1e-6
    # absoulte tolerance for ode solver
    setup.atol = 1e-6

    # Physical parameters
    # Central wavelength [nm]
    setup.wavelengths = [835, 835, 835]
    # Nonlinear coefficients [1/W/m]
    setup.gamma = {
            'gamma11': 0.11,
            'gamma33': 0.11,
            'gamma44': 0.11,
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

    setup.deltabeta = 0
    # Dispersion: derivatives of propagation constant at central wavelength
    # n derivatives of betas are in [ps^n/m]
    betas = np.array([-11.830e-3])
    # Input pulse: pulse duration [ps]
    tFWHM = 0.050
    # for dispersive length calculation
    t0 = tFWHM / 2 / np.log(1 + np.sqrt(2))

    # 3rd order soliton conditions
    ###########################################################################
    # Dispersive length
    LD = t0 ** 2 / np.abs(betas[0])
    # Non-linear length for 3rd order soliton
    LNL = LD / (3 ** 2)
    # Input pulse: peak power [W]
    power = 1 / (LNL * setup.gamma['gamma11'])
    # Length of soliton, in which it break dispersive characteristic
    Z0 = np.pi * LD / 2
    # Fiber length [m]
    setup.fiber_length = .5
    # Type of pulse:  hyperbolic secant
    setup.pulse_model = [None] * setup.num_channels
    setup.pulse_model[0] = gnlse.CWEnvelope(0)
    setup.pulse_model[1] = gnlse.CWEnvelope(0)
    setup.pulse_model[2] = gnlse.SechEnvelope(power, 0.050)
    # Loss coefficient [dB/m]
    loss = 0
    # Type of dyspersion operator: build from Taylor expansion
    setup.dispersion_models = gnlse.dispersion.TripleDispersionFiberFromTaylor(loss, betas)

    # Set type of Raman scattering function and selfsteepening
    simulation_type = {
        '3rd order soliton': (False, None),
        '3rd order soliton\nwith self-steepening': (True, None),
        'Raman induced fission\nof 3rd order soliton': (True,
                                                        gnlse.raman_blowwood)
    }

    count = len(simulation_type)
    plt.figure(figsize=(15, 7), facecolor='w', edgecolor='k')
    for (i, (name,
             (self_steepening,
              raman_model))) in enumerate(simulation_type.items()):
        setup.raman_model = raman_model
        setup.self_steepening = self_steepening
        solver = mm_gnlse.MMGNLSE(setup)
        solution = solver.run()

        # modify certain terms in solution to make the plots work
        solution.W = solution.W[2, :]
        solution.AW = solution.AW[:, 2, :]
        solution.At = solution.At[:, 2, :]

        plt.subplot(2, count, i + 1)
        plt.title(name)
        gnlse.plot_wavelength_vs_distance(solution, WL_range=[400, 1400])

        plt.subplot(2, count, i + 1 + count)
        gnlse.plot_delay_vs_distance(solution, time_range=[-.25, .25])

    plt.tight_layout()
    plt.show()
