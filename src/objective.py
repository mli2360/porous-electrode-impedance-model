import pandas as pd
import numpy as np

from src.electrode import calculate_porous_electrode_impedance
from src.particle_network import pull_particle_network_relevant_numbers


def generate_naive_guess(n_particles, spatial_variations, legendre_basis):
    """
    Generates a naive initial guess for the parameter vector and the
    characteristic change in the parameter vector for optimizing a model
    involving multiple particles.

    Parameters
    ----------
    n_particles : int
        The number of particles to be simulated in the model.
    spatial_variations : list of bool
        A list indicating whether there is a spatial variation for each
        parameter.
    legendre_basis : int
        The number of Legendre polynomials used to describe spatial
        variations.

    Returns
    -------
    x0 : numpy.ndarray
        The initial guess for the parameter vector.
    del_x0 : numpy.ndarray
        The characteristic change in the parameter vector to be used in
        optimization.

    Notes
    -----
    The function initializes the guess for various model parameters including
    electrode parameters (resistances and time constants), connection
    probabilities, network parameters, and particle parameters. It also
    considers spatial variations and adjusts the initial guesses accordingly
    based on the presence of spatial variations indicated by the
    spatial_variations parameter list.
    """
    
    # relevant constants
    _, _, n_variables = pull_particle_network_relevant_numbers(n_particles)

    # Initialize arrays for initial guess and characteristic changes
    x0, del_x0 = np.array([]), np.array([])

    # Characteristic electrode parameters (lnR_bulk, lnR_lyte_ion,
    # lnR_lyte_electron, lnt_lyte)
    x0 = np.append(x0, np.zeros(4))  # Initial guess
    del_x0 = np.append(del_x0, np.full(4, 0.25))  # Characteristic changes

    # Connection probabilities and network parameters
    x0 = np.append(
        x0, np.concatenate((np.array([0.05]), np.full(n_variables-1, 0.5), np.zeros(n_variables)))
    )
    del_x0 = np.append(
        del_x0, np.concatenate((np.full(n_variables, 0.025), np.full(n_variables, 0.25)))
    )

    # Particle parameters and spatial variations
    for idx in range(n_particles):
        # Basic particle parameters: R_ct, C_dl, lnL_part, lnD_part
        x0 = np.append(x0, [0, 0, -14.5, -33.6])
        del_x0 = np.append(del_x0, np.full(4, 0.25))

        # Spatial variations for particle parameters
        if spatial_variations[idx]:
            x0 = np.append(x0, np.zeros(legendre_basis))
            del_x0 = np.append(del_x0, np.full(legendre_basis, 0.5))

    # Network parameters coefficients, considering spatial variations
    for idx in range(n_variables):
        if spatial_variations[n_particles + idx]:
            x0 = np.append(x0, np.zeros(legendre_basis))
            del_x0 = np.append(del_x0, np.full(legendre_basis, 0.5))

    sv_idx = n_particles + n_variables
    # Particle parameters coefficients, considering spatial variations
    for idx in range(n_particles):
        for _ in ['lnR_ct', 'lnC_dl', 'lnL_part']:  # Looping but not using fields directly
            if spatial_variations[sv_idx]:
                x0 = np.append(x0, np.zeros(legendre_basis))
                del_x0 = np.append(del_x0, np.full(legendre_basis, 0.5))
            sv_idx += 1
            
        # Spatial variations for Legendre polynomial coefficients of coefficients
        for _ in range(legendre_basis):  # Looping but not directly using index
            if spatial_variations[sv_idx]:
                x0 = np.append(x0, np.zeros(legendre_basis))
                del_x0 = np.append(del_x0, np.full(legendre_basis, 0.5))
            sv_idx += 1

    return x0, del_x0


def unpack_parameters(x, data, n_particles, bool_spatial_variations,
                      n_basis=6):
    """
    Unpack parameters from a linear array and experimental data into a
    structured dictionary.

    This function unpacks a flat array of parameters and experimental data into
    a structured dictionary suitable for further processing, particularly for
    calculating the objective function in porous electrode modeling.

    Parameters
    ----------
    x : numpy.ndarray
        Linear array of parameters defining the model.
    data : pandas.DataFrame
        Experimental data containing frequency and impedance measurements.
    n_particles : int
        Number of particles simulated within the model.
    bool_spatial_variations : dict of {str: list of bool}
        Dictionary indicating spatial variations for particle and network parameters.
    n_basis : int, optional
        Number of basis fucntions for all spatially-defined functions (default
        is 6).

    Returns
    -------
    dict
        Dictionary of unpacked parameters structured for model simulation, including
        electrode and particle network properties and any spatial variation descriptors.

    Notes
    -----
    This function is tailored to unpack parameters for simulations of porous electrode
    models, especially handling spatial variations with Legendre polynomial descriptions.
    It assumes specific naming and structuring conventions for input data and parameters.
    """
    
    # Constants derived from the number of particles
    _, _, n_variables = pull_particle_network_relevant_numbers(n_particles)

    # Unpack experimental data frequencies
    w = data['freq/Hz'].values

    # Initialize the parameters dictionary
    parameters = {
        'w': w,
        # Unpacking electrode parameters
        'R_bulk': np.exp(x[0]),
        'R_electrolyte_ion': np.exp(x[1]),
        'R_electrolyte_electron': np.exp(x[2]),
        'C_electrolyte': np.exp(x[3]),
        # Connection parameters
        'connectivity_source_target': x[4:4 + n_particles],
        'connectivity_drain_target': x[4 + n_particles:4 + 2 * n_particles],
        'connectivity_target_target': x[4 + 2 * n_particles:4 + n_variables],
        # Resistance parameters
        'R_source_target': np.exp(x[4 + n_variables: 4 + n_variables + n_particles]),
        'R_drain_target': np.exp(x[4 + n_variables + n_particles:4 + n_variables + 2 * n_particles]),
        'R_target_target': np.exp(x[4 + n_variables + 2 * n_particles:4 + 2 * n_variables]),
        # Placeholder for coefficients, to be filled below
        'R_source_target_coefficients': [None] * n_particles,
        'R_drain_target_coefficients': [None] * n_particles,
        'R_target_target_coefficients': [None] * (n_particles * (n_particles - 1)),
        'particle_params': [{}] * n_particles,
        'particle_params_coefficients': [{}] * n_particles,
    }

    # Start index for unpacking the parameters
    p_idx = 4 + 2 * n_variables

    # Unpack coefficients of resistance in the particle network if spatial
    # variations exist
    for field in ['R_source_target', 'R_drain_target', 'R_target_target']:
        if bool_spatial_variations["Electrode"].get(field.lower(), False):
            for idx in range(n_particles):
                parameters[field + '_coefficients'][idx] = x[p_idx:p_idx + n_basis]
                p_idx += n_basis

    # Unpack particle parameters and their spatial variations
    for idx in range(n_particles):
        particle = {}  # Temporary storage for particle-specific parameters
        particle_coeffs = {"Electrode" : {}, "Particle" : {}}  # Temporary storage for particle-specific coefficient variations

        # Standard particle parameters
        for field in ['R_ct', 'C_dl', 'L_part', 'D_part']:
            particle[field] = np.exp(x[p_idx])
            p_idx += 1

        # Coefficients for spatial variations on the particle scale
        if bool_spatial_variations["Particle"].get('d_part', False):
            particle_coeffs["Particle"]["D_part_coefficients"] = x[p_idx:p_idx + n_basis]
            p_idx += n_basis
        else:
            particle_coeffs["Particle"]["D_part_coefficients"] = None

        # Spatial variations on the electrode scale
        for field in ['R_ct', 'C_dl', 'L_part']:
            if bool_spatial_variations["Electrode"].get(field.lower(), False):
                particle_coeffs["Electrode"][field + '_coefficients'] = x[p_idx:p_idx + n_basis]
                p_idx += n_basis
            else:
                particle_coeffs["Electrode"][field + '_coefficients'] = None
        for field in ['D_part']:
            if bool_spatial_variations["Electrode"].get(field.lower(), False):
                if bool_spatial_variations["Particle"].get(field.lower(), False):
                    coeff_list = [None] * n_basis
                    for idx2 in range(n_basis):
                        coeff_list[idx2] = x[p_idx:p_idx + n_basis]
                        p_idx += n_basis
                    particle_coeffs["Electrode"][field + '_coefficients'] = coeff_list
                else:
                    particle_coeffs["Electrode"][field + '_coefficients'] = x[p_idx:p_idx + n_basis]
                    p_idx += n_basis
            else:
                particle_coeffs["Electrode"][field + '_coefficients'] = None

        # Update the main parameters dictionary
        parameters["particle_params"][idx] = particle
        parameters["particle_params_coefficients"][idx] = particle_coeffs

    return parameters


def calculate_misfit_residual(measured_eis, predicted_eis, setting):
    """
    Calculates the misfit residual between measured and predicted EIS datasets.

    Parameters
    ----------
    measured_eis : pandas.DataFrame
        The measured dataset containing columns "freq/Hz", "Re(Z)/Ohm", and
        "-Im(Z)/Ohm".
    predicted_eis : pandas.DataFrame
        The predicted dataset containing columns "freq/Hz", "Re(Z)/Ohm", and
        "-Im(Z)/Ohm".
    setting : str
        Specifies the calculation mode: "ReOnly", "ImOnly", or "ReIm" to
        consider  the real part, imaginary part, or both parts of the impedance,
        respectively.

    Returns
    -------
    float
        The normalized least square error based on the specified setting,
        averaged across all data points.

    Raises
    ------
    ValueError
        If the "freq/Hz" values in the measured and predicted datasets do not
        match.
    """
    
    # Ensure frequency values match between datasets
    if not np.all(np.isclose(measured_eis['freq/Hz'], predicted_eis['freq/Hz'])):
        raise ValueError('The "freq/Hz" values in the datasets do not match.')
    
    # Initialize misfit residual
    misfit_residual = 0
    
    # Split settings into error setting and normalization setting
    setting_split = setting.split("-")
    error_setting = setting_split[0]
    if len(setting_split) == 2:
        normalization_setting = setting_split[1]
    else:
        normalization_setting = None
    
    # Calculate normalization factor to avoid division by zero
    normalization = np.sqrt(measured_eis['Re(Z)/Ohm']**2
                            + measured_eis['-Im(Z)/Ohm']**2)
    if normalization_setting is not None:
        normalization_correction = np.maximum(
            (np.log(np.abs(measured_eis['Re(Z)/Ohm']))
            + np.log(np.abs(measured_eis['-Im(Z)/Ohm']))), 1)
        normalization /= normalization_correction

    # Ensure no division by zero
    normalization = np.where(normalization == 0, np.nan, normalization)
    
    # Calculate least square error based on setting
    if error_setting in ["ReOnly", "ImOnly", "ReIm"]:
        errors = []
        if error_setting in ["ReOnly", "ReIm"]:
            errors.append((measured_eis['Re(Z)/Ohm'] - predicted_eis['Re(Z)/Ohm']) 
                          / normalization)
        if error_setting in ["ImOnly", "ReIm"]:
            errors.append((measured_eis['-Im(Z)/Ohm'] - predicted_eis['-Im(Z)/Ohm'])
                          / normalization)
        
        # Combine errors and calculate RMS
        error = np.hstack(errors)
        misfit_residual = np.nan_to_num(misfit_residual, nan=1e2)
        misfit_residual = np.sqrt(np.mean(error**2))

    else:
        raise ValueError('Invalid setting. Choose "ReOnly", "ImOnly", or '
                         + '"ReIm".')

    # Handle potential NaN values in misfit calculation
    misfit_residual = np.nan_to_num(misfit_residual, nan=1e20)

    return misfit_residual


def calc_norm_complexity(params, setting, scale=None, expected_values=None,
                         weights=None):
    """
    Calculates the complexity of a parameter set based on a specified setting,
    optionally adjusting parameters by scaling and shifting.

    Parameters
    ----------
    params : list or numpy.array
        The list of parameter values to be evaluated.
    setting : str
        Specifies the complexity measure: "L1" for L1 norm, "L2" for L2 norm,
        or "ElasticNet" for a weighted combination of L1 and L2 norms.
    scale : list or numpy.array, optional
        Scale factors for each parameter, must be of the same length as
        `params`.
    expected_values : list or numpy.array, optional
        Expected values to be subtracted from `params`, must be of the same
        length as `params`.
    weights : list or numpy.array, optional
        Weights for combining L1 and L2 norms in "ElasticNet" setting, must be
        of length 2 and sum to 1.

    Returns
    -------
    float
        The calculated complexity measure.

    Raises
    ------
    ValueError
        If `scale` or `expected_values` do not match `params` in length, or
        if `weights` is invalid for the "ElasticNet" setting.
    """
    
    # Convert parameters to numpy array for efficient computation
    params = np.array(params)
    
    # Validate and apply expected_values shift
    if expected_values is not None:
        if len(expected_values) != len(params):
            raise ValueError(f"Length of 'expected_values' {len(expected_values)}"
                             + f" does not match 'params' {len(params)}.")
        params -= np.array(expected_values)
 
    # Validate and apply scale
    if scale is not None:
        if len(scale) != len(params):
            raise ValueError("Length of 'scale' does not match 'params'.")
        params /= np.array(scale)
    
    # Compute complexity based on the specified setting
    complexity = 0
    if setting == "L1":
        complexity = np.sum(np.abs(params))
    elif setting == "L2":
        complexity = np.sqrt(np.sum(np.square(params)))
    elif setting == "ElasticNet":
        if weights is None or len(weights) != 2 or not np.isclose(np.sum(weights), 1):
            raise ValueError("Invalid 'weights' for 'ElasticNet': Must be of "
                             + "length 2 and sum to 1.")
        complexity = (weights[0] * np.sum(np.abs(params))
                      + weights[1] * np.sqrt(np.sum(np.square(params))))
    else:
        raise ValueError("Invalid 'setting': Choose 'L1', 'L2', or "
                         + "'ElasticNet'.")

    return complexity


def calculate_legendre_polynomial_complexity(coefficients, weights):
    """
    Calculate the weighted complexity of a Legendre polynomial.

    The complexity is defined as a weighted sum of the L1 norm (sum of absolute
    values of the coefficients) and the L2 norm (square root of the sum of 
    squares of the coefficients). The weights for the L1 and L2 norms must add 
    up to 1.

    Parameters
    ----------
    coefficients : array_like
        Coefficients of the Legendre polynomial.
    weights : array_like
        Weights for the L1 and L2 norms. Should contain exactly two elements 
        that sum up to 1.

    Returns
    -------
    complexity : float
        The calculated weighted complexity of the Legendre polynomial.

    Raises
    ------
    ValueError
        If the weights do not contain exactly two elements or do not sum to 1.

    Examples
    --------
    >>> calculate_legendre_polynomial_complexity([1, -0.5, 0.25], [0.5, 0.5])
    1.118033988749895
    """
    
    if len(weights) != 2 or not np.isclose(np.sum(weights), 1):
        raise ValueError('Weights must consist of two elements that sum to 1.')

    # Calculate the weighted complexity using the L1 and L2 norms
    complexity = (
        weights[0] * np.linalg.norm(coefficients, 1)  # L1 norm
        + weights[1] * np.linalg.norm(coefficients, 2)  # L2 norm
    )
    
    return complexity / coefficients.size


def calculate_connectivity_complexity(connection_source_target,
                                      connection_drain_target,
                                      connection_target_target):
    """
    Calculates the normalized complexity of network connectivity based on the 
    connection probabilities between different components within a particle 
    network. This function evaluates the degree of connectivity and interaction 
    between various network components by transforming their connection 
    probabilities into a measure of complexity.

    Parameters
    ----------
    connection_source_target : numpy.ndarray
        Array containing the connection probabilities from source nodes to 
        target nodes within the network.
    connection_drain_target : numpy.ndarray
        Array containing the connection probabilities from drain nodes to 
        target nodes within the network.
    connection_target_target : numpy.ndarray
        Array containing the connection probabilities between target nodes 
        within the network.

    Returns
    -------
    float
        The overall complexity of the network connections, calculated as the 
        sum of individual complexities normalized by the total number of 
        connection types.

    Notes
    -----
    The complexity is calculated by assessing different aspects:
    - Normalized complexity for binary connections between nodes
    - Ratio of interparticle to particle-backbone connections
    - Circular complexity, evaluating the circular connections between particles

    The function ensures that the complexity reflects variations in network 
    topology and connectivity between particles.
    """
    
    # Combine all connection probabilities into a single array
    connections = np.concatenate([connection_source_target,
                                  connection_drain_target,
                                  connection_target_target])

    # Initial complexity calculation based on negative cosine transformation
    complexity_0_or_1 = np.sum(-np.cos(2 * np.pi * connections) + 1)
    n_connections = (len(connection_source_target)
                     + len(connection_drain_target)
                     + len(connection_target_target))
    complexity_0_or_1 /= n_connections

    # Interparticle complexity contrasting connections between particles and to
    # the backbone
    complexity_interparticle = np.exp(np.sum(connection_target_target) /
        (np.sum(connection_source_target)+np.sum(connection_drain_target))) - 1

    # Define the total number of particles for circular complexity calculation
    n_particles = len(connection_source_target)
    complexity_circular = 0  # Initial circular complexity
    already_compared = []  # Keeping track of compared index pairs to avoid redundancy

    if n_particles >= 2:
        # Calculate circular complexity based on paired connection probabilities
        for idx in range(len(connection_target_target)):
            if idx not in already_compared:
                # Index decompositions to identify particle pairs
                particle_curr = idx // (n_particles-1)
                particle_pair = idx % (n_particles-1)
                if particle_pair >= particle_curr:
                    particle_pair += 1  # Adjusting for zero-based indexing

                # Cross-index for symmetric connection
                idx_other = particle_pair * (n_particles - 1) + particle_curr
                epsilon = 1e-2  # Small value to avoid division by zero

                # Circular complexity calculation for particle pairs
                complexity_circular += (100
                    / (1 + np.exp(-5*(connection_target_target[idx] - 1)))
                    / (1 + np.exp(-5*(connection_target_target[idx_other] - 1)))
                )

                # Marking indices as compared
                already_compared.extend([idx, idx_other])
        
        # Final normalization of circular complexity
        # Adjusting for total potential connections
        complexity_circular /= (n_particles * (n_particles - 1) / 2)
    
    # Combine complexities for total measure
    total_complexity = complexity_0_or_1 + complexity_interparticle + complexity_circular

    return total_complexity


def calculate_spatial_function_complexity(function_type, n_basis,
        source_target_coeffs, drain_target_coeffs, target_target_coeffs,
        particle_params_coeffs, elastic_net_weights, scaling
):
    """
    Calculates the complexity of spatially varying parameters based on Legendre
    polynomial coefficients.

    This function computes the complexity of spatial variations of parameters
    within a porous electrode model. Complexity is evaluated as the
    regularization of coefficients for Legendre polynomials that describe
    spatial variations. A model with no spatially varying parameters is
    considered to have zero complexity. The function accounts for the complexity
    due to spatial variation in diffusivity parameters within particles and
    resistance parameters within the particle network.

    Parameters
    ----------
    particle_params : dict
        Dictionary containing the particle parameters including the coefficients
        for spatial variations.
    electrode_params_coefficients : list of numpy.ndarray
        List containing arrays of coefficients for spatial variations in the
        network parameters.
    particle_params_coefficients : list of dict
        List of dictionaries, each containing arrays of coefficients for
        spatial variations in individual particle parameters.

    Returns
    -------
    float
        The normalized complexity of the spatially varying parameters within the
        model.

    Notes
    -----
    The complexity is normalized by the number of spatial functions and adjusted
    for the logarithmic scaling of parameters. The function assumes equal weight
    for all types of spatial variations and applies a normalization factor to
    account for potential changes in the parameters by orders of magnitude.
    """
    
    complexity = 0
    n_spatial_functions = 0

    if function_type == 'legendre':
        function_complexity = calculate_legendre_polynomial_complexity
    else:
        raise ValueError('unknonwn function type')
    
    # Calculate complexity due to spatial variation in the particle network 
    # resistances
    for coeffs in source_target_coeffs:
        if coeffs is not None:
            complexity += function_complexity(coeffs, elastic_net_weights)
            n_spatial_functions += 1
    for coeffs in drain_target_coeffs:
        if coeffs is not None:
            complexity += function_complexity(coeffs, elastic_net_weights)
            n_spatial_functions += 1
    for coeffs in target_target_coeffs:
        if coeffs is not None:
            complexity += function_complexity(coeffs, elastic_net_weights)
            n_spatial_functions += 1


    # Calculate complexity due to spatial variation in the parameters at the
    # particle scale
    for particle_coeffs in particle_params_coeffs:
        if "Particle" in particle_coeffs.items():
            if "D_part" in particle_coeffs["Particle"]:
                coeffs = particle_coeffs["Particle"]["D_part"]
                complexity += function_complexity(coeffs, elastic_net_weights)
                n_spatial_functions += 1
        elif "Electrode" in particle_coeffs.items():
            for field in ["R_ct", "C_dl", "L_part", "D_part"]:
                if field in particle_coeffs["Electrode"]:
                    coeffs = particle_coeffs["Electrode"][field]
                    if isinstance(coeffs[0], list):
                        for coeff in coeffs:
                            complexity += function_complexity(coeff, elastic_net_weights)
                            n_spatial_functions += 1
                    else:
                        complexity += function_complexity(coeffs, elastic_net_weights)
                        n_spatial_functions += 1

    # Normalize complexity and adjust for logarithmic scaling
    if n_spatial_functions > 0:
        return complexity / (n_spatial_functions * scaling)
    else:
        return 0  # Return zero complexity if there are no spatial functions


def calculate_parameter_values_complexity(parameters, weights, settings):
    """
    Calculates the overall complexity of parameter values based on their
    deviation from expected physical ranges.

    Parameters
    ----------
    parameters : dict
        Dictionary containing parameter values for the model, including
        connection probabilities, electrolyte properties, and particle parameters.
    weights : list or numpy.ndarray
        Weights applied to different components of complexity. Should sum to 1.
    settings : dict
        Dictionary containing physical bounds and temperatures for evaluating
        complexities.

    Returns
    -------
    float
        The calculated overall complexity based on parameter values.

    Raises
    ------
    ValueError
        If the sum of weights does not equal 1 or the length of weights does
        not match the number of evaluated complexities.

    Notes
    -----
    This function calculates the complexity of the model parameters to penalize
    configurations that deviate significantly from expected physical ranges.
    The complexity is calculated across several aspects including connection
    probabilities, time scales, and resistances, each weighted according to
    `weights`. Parameters should lie within typical physical ranges to minimize
    complexity.
    """

    complexities = []
    
    # Connection probabilities complexity
    temperature = settings["connectivity_bounds_temperature"]
    connections_complexity = (
        np.sum(np.exp((-0.0 - parameters['connectivity_source_target']) / temperature))
        + np.sum(np.exp((parameters['connectivity_source_target'] - 1.1) / temperature))
        + np.sum(np.exp((-0.0 - parameters['connectivity_drain_target']) / temperature))
        + np.sum(np.exp((parameters['connectivity_drain_target'] - 1.1) / temperature))
        + np.sum(np.exp((-0.0 - parameters['connectivity_target_target']) / temperature))
        + np.sum(np.exp((parameters['connectivity_target_target'] - 1.1) / temperature))
    )
    n_variables = (
        len(parameters['connectivity_source_target'])
        + len(parameters['connectivity_drain_target'])
        + len(parameters['connectivity_target_target'])
    )
    complexities.append(connections_complexity / n_variables)


    # Time scales complexity
    ln_t_lyte = np.log((parameters["C_electrolyte"]
        * (parameters["R_electrolyte_ion"] + parameters["R_electrolyte_electron"])))
    ln_t_min, ln_t_max = \
        np.log(settings["t_lyte_min"]), np.log(settings["t_lyte_max"])
    temperature = settings["t_lyte_bounds_temperature"]
    time_complexity = (np.exp((ln_t_lyte - ln_t_max) / temperature)
                       + np.exp((ln_t_min - ln_t_lyte) / temperature))
    n_complexities = 1

    for particle in parameters['particle_params']:
        # Determine the reaction and diffusion time scales for each particle
        ln_t_rxn = np.log(particle['R_ct'] * particle['C_dl'])
        ln_t_diff = np.log(particle['L_part']**2 / particle['D_part'])
        # Pull the min, max bounds and bounds temperature
        ln_t_rxn_min, ln_t_rxn_max, ln_t_diff_min, ln_t_diff_max = \
            np.log(settings["t_rxn_min"]), np.log(settings["t_rxn_max"]), \
            np.log(settings["t_diff_min"]), np.log(settings["t_diff_max"])
        T_rxn, T_diff = (settings["t_rxn_bounds_temperature"],
                         settings["t_diff_bounds_temperature"])
        # calculate complexity for both timescales
        time_complexity += (np.exp((ln_t_rxn - ln_t_rxn_max) / T_rxn)
                            + np.exp((ln_t_rxn_min - ln_t_rxn) / T_rxn))
        time_complexity += (np.exp((ln_t_diff - ln_t_diff_max) / T_diff)
                            + np.exp((ln_t_diff_min - ln_t_diff) / T_diff))
        n_complexities += 2

    complexities.append(time_complexity / n_complexities)

    # Resistance complexity
    n_complexities = 0
    # Electrode-scale resistances
    ln_R_min, ln_R_max, temperature = (
        np.log(settings["r_lyte_min"]), np.log(settings["r_lyte_max"]), 
        settings["r_lyte_bounds_temperature"])
    ln_R_lyte = np.log([parameters['R_electrolyte_ion'],
                        parameters['R_electrolyte_electron'],
                        parameters['R_bulk']])
    resistance_complexity = np.sum(np.exp((ln_R_lyte - ln_R_max) / temperature)
                                   + np.exp((ln_R_min - ln_R_lyte) / temperature))
    n_complexities = 3

    # Network-scale resistances
    ln_R_network = np.log(np.concatenate([parameters["R_source_target"],
                                         parameters["R_drain_target"],
                                         parameters["R_target_target"]]))
    ln_R_min, ln_R_max, temperature = (
        np.log(settings["r_network_min"]), np.log(settings["r_network_max"]), 
        settings["r_network_bounds_temperature"])
    resistance_complexity += np.sum(
        np.exp((ln_R_network - ln_R_max) / temperature)
        + np.exp((ln_R_min - ln_R_network) / temperature))
    n_complexities += n_variables

    # Particle-scale resistances
    ln_R_ct = np.log([particle["R_ct"] for particle in parameters["particle_params"]])
    ln_R_min, ln_R_max, temperature = (
        np.log(settings["r_ct_min"]), np.log(settings["r_ct_max"]),
        settings["r_ct_bounds_temperature"])
    resistance_complexity += np.sum(np.exp((ln_R_ct - ln_R_max) / temperature)
                                    + np.exp((ln_R_min - ln_R_ct) / temperature))
    n_complexities += len(parameters["particle_params"])

    # Particle Lengths
    ln_L_part = np.log([particle["L_part"] for particle in parameters["particle_params"]])
    ln_L_part_min, ln_L_part_max, temperature = (
        np.log(settings["l_part_min"]), np.log(settings["l_part_max"]),
        settings["l_part_bounds_temperature"])
    resistance_complexity += np.sum(np.exp((ln_L_part - ln_L_part_max) / temperature)
                                    + np.exp((ln_L_part_min - ln_L_part) / temperature))
    n_complexities += len(parameters["particle_params"])

    complexities.append(resistance_complexity / n_complexities)

    # Validate weights
    if not np.isclose(np.sum(weights), 1):
        raise ValueError("Weights must sum to 1.")
    if len(weights) != len(complexities):
        raise ValueError("Length of weights must match the number of complexity"
                         " components.")

    # Calculate total complexity
    complexity = np.dot(complexities, weights)

    return complexity


def calculate_objective_function(x, data, n_particles, objective_weights,
                                 bool_spatial_variations,
                                 n_basis=6, inputs=None):
    """
    Calculates the objective function for fitting a porous-electrode-based EIS
    model.

    This function calculates the objective function for the model fit of a
    porous-electrode-based electrochemical impedance spectroscopy (EIS) model,
    where each volume element is defined by multiple simulated particles
    connected in a network structure. The fit of the model to experimental data
    is assessed, alongside the model's complexity, which is modulated by
    `objective_weights`. The model allows for spatial variations in parameters,
    enhancing the realism of the simulation.

    Parameters
    ----------
    x : numpy.ndarray
        The array of model parameters.
    data : dict
        Dictionary containing the frequency ('freq/Hz') and impedance data
        ('Re(Z)/Ohm', '-Im(Z)/Ohm').
    n_particles : int
        The number of particles simulated within the electrode model.
    objective_weights : numpy.ndarray
        Weights for different components of the objective function, should sum
        up to 1.
    spatial_variations : list of bool
        Indicates whether spatial variations are considered for the
        corresponding model parameters.
    legendre_basis : int, optional
        The number of Legendre polynomials used for describing spatial-dependent
        functions, by default 6.

    Returns
    -------
    float
        The calculated objective function value, incorporating both fit to
        experimental data and model complexity.

    Raises
    ------
    ValueError
        If the sum of all objective weights does not equal 1 or if the number
        of weights does not match the number of calculated objectives.

    Notes
    -----
    The objective function is a weighted sum of different metrics: misfit
    between experimental and predicted data, complexity of the network
    connectivity, complexity arising from spatial variation in parameters, 
    and complexity associated with physical constraints on parameter values.
    """
    
    # relevant constants
    n_nodes, n_edges, n_variables = \
        pull_particle_network_relevant_numbers(n_particles)
    
    # Unpack parameters and data
    parameters = unpack_parameters(x, data, n_particles,
                                   bool_spatial_variations, n_basis)
    
    # Calculate predicted impedance and construct dataframes for comparison
    measured_eis = data
    Z = calculate_porous_electrode_impedance(
        **parameters,n_samples=inputs["Electrode"]["n_samples"], inputs=inputs
    )
    predicted_eis = pd.DataFrame({
        'freq/Hz': parameters['w'],  # frequency
        'Re(Z)/Ohm': Z.real,         # Real part of impedance
        '-Im(Z)/Ohm': -Z.imag,       # Negative imaginary part of impedance
    })
    
    # Initialize objective measures
    objectives = []

    # Misfit is a measure how well the measured and predicted datasets match.
    # Here, it is quantified by looking at the error between the real and
    # imaginary components of impedance evaluated at the same frequencies. This
    # should be one of the most important / heavily weighted objectives.
    objective_misfit = calculate_misfit_residual(
        measured_eis, predicted_eis, setting=inputs["Objective"]["misfit_setting"]
    )
    objectives.append(objective_misfit)

    # Complexity is associated with the connectivity structure in the
    # particle network. In the case of there exists connections between all
    # particles, we could intuit that the model has a high complexity.
    objective_connectivity = calculate_connectivity_complexity(
        parameters['connectivity_source_target'],
        parameters['connectivity_drain_target'],
        parameters['connectivity_target_target'],
    )
    objectives.append(objective_connectivity)
    
    # Complexity is associated with the coefficients governing each of the
    # space-dependent functions. Complexity is assumed to be small when the 
    # parameters are not dependent on space.
    objective_spatial_function = calculate_spatial_function_complexity(
        inputs["Function"]["type"],
        inputs["Function"]["n_basis"],
        parameters['R_source_target_coefficients'],
        parameters['R_drain_target_coefficients'],
        parameters['R_target_target_coefficients'],
        parameters['particle_params_coefficients'],
        inputs['Objective']['basis_function_elastic_weights'],
        inputs['Objective']['spatial_basis_scale'],
    )
    objectives.append(objective_spatial_function)

    # Complexity is associated with a few of the physical variables. If
    # known variables are above or below a certain value. One set of key values
    # are the diffusion time scale of the intercalated lithium ions and the 
    # diffusion time scale of the electrolyte ions. Both are known to be within
    # a wide range, but a significantly constrained region relative to the full
    # parameter range
    objective_parameter_values = calculate_parameter_values_complexity(
        parameters, inputs["Objective"]["parameter_complexity_weights"],
        settings=inputs["Objective"]
    )
    objectives.append(objective_parameter_values)
    
    # Validate the input of objective_weights
    if not np.isclose(np.sum(objective_weights), 1):
        raise ValueError("The sum of all the weightings for relevant objectives"
                         + "should be close to 1")
    if len(objective_weights) != 4:
        raise ValueError("The number of objectives weights needs to be the same"
                         "as the number of objectives calculated")
    
    # Calculate the total objective value
    objective = 0
    for idx, obj in enumerate(objectives):
        objective += obj * objective_weights[idx]
    
    return objective, predicted_eis

