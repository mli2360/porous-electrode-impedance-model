import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

from src.constructions import impedance_transmission_line
from src.particle_network import (calculate_particle_network_impedance,
                              generate_connectivity_matrix,
                              pull_particle_network_relevant_numbers)
from src.functions import evaluate_parameters


def calculate_electrode_volume_impedance(w, C_lyte, n_particles,
                                         R_network, connectivity_network,
                                         R_ct, C_dl, L_part, D_part,
                                         particle_params,
                                         particle_params_coefficients,
                                         inputs=None):
    """
    Calculates the impedance of an electrode volume in a porous electrode model.

    This function considers both the macroscopic properties of the electrode,
    such as electrolyte capacitance and particle network resistances, and the
    microscopic properties of individual particles, such as charge transfer and
    diffusion resistances, to calculate the overall impedance of a discrete
    volume element of the electrode.

    Parameters:
    -----------
    w : numpy.ndarray
        An array of angular frequencies at which to calculate the impedance.
    C_lyte : float
        Capacitance of the electrolyte within the electrode volume.
    n_particles : int
        Number of active material particles in the electrode volume.
    R_network : numpy.ndarray
        Resistances of the electrical connections between particles in the network.
    connectivity_network : numpy.ndarray
        Connectivity between particles within the electrode volume.
    R_ct : list
        Charge transfer resistance of each particle.
    C_dl : list
        Double layer capacitance of each particle.
    L_part : list
        Characteristic length of each particle.
    D_part : list
        Diffusion coefficient of each particle.
    particle_params : list of dicts
        List containing parameter dictionaries for each particle.
    particle_params_coefficients : list of dicts
        List containing dictionaries of coefficients for spatial variations of
        particle parameters.
    inputs : dict, optional
        Additional parameters required for the calculation, if any.

    Returns:
    --------
    numpy.ndarray
        The calculated impedance of the electrode volume at the specified frequencies.

    Notes:
    ------
    This function integrates over the contributions from individual particles and the
    connectivity of the particle network to derive the impedance of the whole
    electrode volume. It accounts for spatial variations if coefficients are provided.
    """
    
    # Generate connectivity matrix for the particle network
    connectivity_matrix = generate_connectivity_matrix(n_particles, 
                                                           connectivity_network)
    
    # Calculate impedance of the particle network within the electrode volume
    Z_particle_network = calculate_particle_network_impedance(
        w, n_particles, R_network, connectivity_matrix,
        R_ct, C_dl, L_part, D_part,
        particle_params, particle_params_coefficients, inputs=inputs
    )
    
    # Account for the pseudo-capacitance of the electrolyte
    Z_pseudocapacitance = 1 / (1j * w * C_lyte)
    
    # Aggregate the impedances to get the total impedance for the electrode
    # volume
    Z_electrode_volume = 1 / (1 / Z_pseudocapacitance + 1 / Z_particle_network)

    # # Create the Nyquist plot
    # plt.figure()
    # plt.plot(Z_electrode_volume.real, -Z_electrode_volume.imag, 'o-', markersize=8)  # 'o-' for line with circle markers
    # plt.xlabel('Re(Z) [Ohms]')
    # plt.ylabel('-Im(Z) [Ohms]')
    # plt.title('Nyquist Plot')
    # plt.grid(True)
    # plt.axis('equal')  # Ensure the plot has equal scaling on both axes

    # # Display the plot
    # plt.draw()  # Draw the current figure
    # plt.pause(5)  # Pause for a few seconds and update plot window  

    return Z_electrode_volume


def calculate_porous_electrode_impedance(
        w,
        R_bulk, R_electrolyte_ion, R_electrolyte_electron, C_electrolyte, 
        connectivity_source_target, connectivity_drain_target, 
        connectivity_target_target, 
        R_source_target, R_drain_target, R_target_target,
        R_source_target_coefficients, R_drain_target_coefficients, 
        R_target_target_coefficients, 
        particle_params, particle_params_coefficients,
        n_samples=100, inputs=None):
    """
    Calculates the impedance of a porous electrode incorporating spatial 
    variations and electrochemical properties.

    This function calculates the impedance based on given electrode and 
    electrochemical properties, including connectivity and resistances of 
    particle networks, as well as particle parameters. Spatial variations are 
    considered if coefficients are provided, enabling the modeling of 
    heterogeneous electrode structures.

    Parameters:
    -----------
    w : numpy.ndarray
        An array of angular frequencies at which to calculate the impedance.
    R_bulk, R_electrolyte_ion, R_electrolyte_electron : float
        Resistances of the bulk electrode, ionic, and electronic components in the 
        electrolyte, respectively.
    C_electrolyte : float
        Capacitance of the electrolyte.
    connectivity_source_target, connectivity_drain_target, 
    connectivity_target_target : numpy.ndarray
        Connectivity probabilities between different network elements.
    R_source_target, R_drain_target, R_target_target : numpy.ndarray
        Resistances between different network elements.
    R_source_target_coefficients, R_drain_target_coefficients, 
    R_target_target_coefficients : list of numpy.ndarray
        Coefficients for spatial variations of network resistances.
    particle_params : list of dicts
        List containing parameter dictionaries for each particle.
    particle_params_coefficients : list of dicts
        List containing dictionaries of coefficients for spatial variations of 
        particle parameters.
    n_samples : int, optional
        Number of spatial discretization samples (default: 100).
    inputs : dict, optional
        Additional input parameters for the function (default: None).

    Returns:
    --------
    numpy.ndarray
        The calculated impedance of the porous electrode at specified frequencies.

    Notes:
    ------
    This function evaluates the impedance based on detailed modeling of 
    electrochemical components and their spatial variations. It integrates 
    contributions from both the microstructure of the particle network and 
    individual particle properties to compute the overall electrode impedance.
    """
    
    # Initialization and parameter extraction
    n_particles = len(particle_params)
    _, _, n_variables = pull_particle_network_relevant_numbers(n_particles)

    # Rescale to Legendre polynomial domain [-1, 1]
    x = np.linspace(1, 0, n_samples) * 2 - 1 

    # Define the series component of the transmission line impedance
    Z_series = (R_electrolyte_ion + R_electrolyte_electron) * \
        np.ones_like(w, dtype=np.cdouble)
    
    # Define the parallel component of the transmission line impedance. This is
    # significantly trickier
    
    # Flag to indicate if there is spatial variation at the electrode scale
    exist_spatial_variation_flag = False

    # Evaluate network and particle parameters at each position along electrode
    R_list = np.concatenate([R_source_target, R_drain_target, R_target_target])
    ln_R_coeffs_list = R_source_target_coefficients + R_drain_target_coefficients \
        + R_target_target_coefficients
    ln_R_list = np.log(R_list)

    # Sample the resistance of the particle network at each position along the electrode 
    R_network_samples = np.zeros((len(R_list), n_samples))
    for idx, (ln_R, coeffs) in enumerate(zip(ln_R_list, ln_R_coeffs_list)):
        if coeffs is not None:
            # we now need to evaluate all the resistance network at each electrode volume, even if it is a constant over space.
            exist_spatial_variation_flag = True
        R_network_samples[idx, :] = \
            np.exp(evaluate_parameters(x, ln_R, coeffs, inputs["Function"]["type"]))
        
    # Sample the connectivity of the particle network at each position along the electrode 
    # Note that there is no spatial variation of the connectivity probabilities, so the sample is the same across the entire electrode
    connectivity_list = np.concatenate([connectivity_source_target,
                                        connectivity_drain_target,
                                        connectivity_target_target])
    connectivity_network_samples = np.tile(connectivity_list, (n_samples, 1)).T
    

    # Sample the particle parameters
    particles_samples = [None] * n_particles
    # construct the crude_packed_particle_samples
    for idx in range(n_particles):
        particle_samples = {"Electrode" : {}, "Particle" : {}}
        particle_values = particle_params[idx]
        particle_coeffs = particle_params_coefficients[idx]
        hierarchy = "Electrode"
        for name, coeffs in particle_coeffs[hierarchy].items():
            field = name.replace("_coefficients", "")
            if field != "D_part": # cannot have coefficients of coefficients so process normally
                y0 = np.log(particle_values[field])
                particle_samples[hierarchy][field] = np.exp(evaluate_parameters(
                    x, y0, coeffs, inputs["Function"]["type"]
                ))
            else: # need to check if the coefficients are coefficients of coefficients
                if isinstance(coeffs, list): 
                    particle_samples[hierarchy][field] = np.zeros((inputs["Functions"]["n_basis"], n_samples))
                    for idx_2, coeff in enumerate(coeffs):
                        y0 = particle_coeffs["Particle"][name][idx_2]
                        particle_samples[hierarchy][field][idx_2, :] = \
                            evaluate_parameters(x, y0, coeff, inputs["Function"]["type"])
                else:
                    y0 = np.log(particle_values[field])
                    particle_samples[hierarchy][field] = np.exp(evaluate_parameters(
                        x, y0, coeffs, inputs["Function"]["type"]
                    ))
        particles_samples[idx] = particle_samples


    # Impedance calculations for each electrode volume element
    Z_parallel = np.zeros((len(w), n_samples), dtype=np.cdouble)
    for idx in range(n_samples):
        # pull the resistance network parameters directly from the sampled array
        # of network parameters
        R_network = R_network_samples[:, idx]

        # pull the particle parameters directly from the sample array of
        # particle parameters stored in each list element and dictionary entry
        connectivity_network = connectivity_network_samples[:, idx]

        # pull the simple parameters for each of the particles
        R_ct = [part["Electrode"]["R_ct"][idx] for part in particles_samples]
        C_dl = [part["Electrode"]["C_dl"][idx] for part in particles_samples]
        L_part = [part["Electrode"]["L_part"][idx] for part in particles_samples]

        # pull the parameters for the particle diffusivty for each of the particles
        if n_particles > 0:
            if particles_samples[0]["Electrode"]["D_part"].ndim == 2:
                D_part = [part["Electrode"]["D_part"][:, idx] for part in particles_samples]
            else:
                D_part = [part["Electrode"]["D_part"][idx] for part in particles_samples]
        
        # Caluclate the impedance of each electrode volume and store as a sample
        # in Z_parallel_x
        Z_parallel[:, idx] = calculate_electrode_volume_impedance(
            w, C_electrolyte, n_particles, 
            R_network, connectivity_network, R_ct, C_dl, L_part, D_part,
            particle_params, particle_params_coefficients,
            inputs=inputs
        )

    # Calculate the overall porous electrode impedance
    Z_porous_electrode = impedance_transmission_line(
        w, n_samples, Z_series, Z_parallel, geometry='planar'
    )
    Z_porous_electrode += R_bulk

    # # Create the Nyquist plot
    # plt.figure()
    # plt.plot(Z_network.real, -Z_network.imag, 'o-', markersize=8)  # 'o-' for line with circle markers
    # plt.xlabel('Re(Z) [Ohms]')
    # plt.ylabel('-Im(Z) [Ohms]')
    # plt.title('Nyquist Plot')
    # plt.grid(True)
    # plt.axis('equal')  # Ensure the plot has equal scaling on both axes

    # # Display the plot
    # plt.draw()  # Draw the current figure
    # plt.pause(5)  # Pause for a few seconds and update plot window    

    # plt.close()

    return Z_porous_electrode
