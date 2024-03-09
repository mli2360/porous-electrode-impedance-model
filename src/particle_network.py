import numpy as np
from numpy.polynomial.legendre import legval
from scipy.sparse import coo_matrix

from src.constructions import impedance_transmission_line
from src.nodal import calculate_equivalent_impedance
from src.functions import evaluate_parameters

import matplotlib.pyplot as plt
import time


def define_particle_network_connectivity_matrix(n_particles, connectivity_bool):
    """
    Generate the network connectivity matrix for a particle system based on the
    provided boolean connectivity array. This matrix is symmetric and represents
    connections between particles and source/drain nodes within the network.

    Parameters
    ----------
    n_particles : int
        The number of particles in the network.
    connectivity_bool : numpy.ndarray
        A boolean array indicating the presence (True) or absence (False) of
        connections in the network. The size of this array must match the
        expected number of editable variables for the given particle network
        size.
    
    Returns
    -------
    scipy.sparse.csr_matrix
        A CSR-format sparse matrix representing the connectivity of the particle
        network.

    Raises
    ------
    ValueError
        If the size of `connectivity_bool` does not match the expected number
        of connections for the given particle network size.
    """

    n_nodes, _, n_variables = pull_particle_network_relevant_numbers(n_particles)

    # Validate the length of the connectivity boolean array
    if len(connectivity_bool) != n_variables:
        raise ValueError(f"Size of `connectivity_bool` ({len(connectivity_bool)})"
                         " does not match the expected number of connections "
                         f"({n_variables}) for a particle network of size {n_particles}.")

    # Initialize row and column indices for non-zero entries in the connectivity
    # matrix
    row_indices, col_indices = np.array([], dtype=int), np.array([], dtype=int)
    
    # Decompose connectivity_bool into specific connectivity types
    source_target_connect = connectivity_bool[:n_particles]
    drain_target_connect = connectivity_bool[n_particles:2*n_particles]
    target_target_connect = connectivity_bool[2*n_particles:]

    # Connections between the source node and the first node of each particle
    row_indices = np.append(row_indices,
                            np.zeros(n_particles)[source_target_connect])
    col_indices = np.append(col_indices,
                            np.arange(1, 1 + n_particles)[source_target_connect])

    # Connections between the drain node and the second node of each particle
    row_indices = np.append(row_indices,
                            np.arange(1 + n_particles, n_nodes - 1)[drain_target_connect])
    col_indices = np.append(col_indices,
                            np.full(n_particles, n_nodes - 1)[drain_target_connect])

    # Inter-particle (target-target) connections
    X, Y = np.meshgrid(np.arange(1, 1 + n_particles),
                       np.arange(1 + n_particles, n_nodes - 1))
    mask = (Y.flatten() - X.flatten()) != n_particles
    row_indices = np.append(row_indices,
                            X.flatten()[mask][target_target_connect])
    col_indices = np.append(col_indices,
                            Y.flatten()[mask][target_target_connect])

    # Connections between the two terminals for every particle should exist
    row_indices = np.append(row_indices,
                            np.arange(1, 1 + n_particles))
    col_indices = np.append(col_indices,
                            np.arange(1 + n_particles, n_nodes - 1))

    # Construct the sparse connectivity matrix, ensuring symmetry
    data = np.ones_like(row_indices, dtype=int)  # Data for non-zero elements
    connectivity_matrix = coo_matrix((data, (row_indices, col_indices)),
                                     shape=(n_nodes, n_nodes))
    connectivity_matrix = (connectivity_matrix + connectivity_matrix.T)

    # Convert to CSR format for efficiency
    return connectivity_matrix.tocsr()


def generate_connectivity_matrix(n_particles, connectivity_weights):
    """
    Generates a random particle network connectivity matrix based on predefined
    connectivity weights.

    This function creates a boolean matrix indicating the presence or absence of
    connections between particles in a network. Each connection's existence is
    determined randomly based on its corresponding weight provided in
    `connectivity_weights`.

    Parameters
    ----------
    n_particles : int
        The number of particles in the network.
    connectivity_weights : numpy.ndarray
        A 1D numpy array containing the probabilities of each possible
        connection within the network being active. The length of this array
        should match the number of possible connections in a fully connected
        network.

    Returns
    -------
    scipy.sparse.coo_matrix
        A boolean matrix representing the connectivity between particles in the
        network. The matrix is square with dimensions `n_particles` x
        `n_particles`, where each element indicates whether a connection between
        two particles exists (True) or not (False).

    Notes
    -----
    The generated network is symmetric since the connection from particle A to
    particle B is indistinguishable from the connection from B to A. The
    diagonal elements of the matrix (representing self-connections) are always
    set to False.
    """
    
    # Generate random boolean values for each potential connection based on their respective weights
    n_connectivity = len(connectivity_weights)
    connectivity_bool = (
        np.random.uniform(size=n_connectivity) < connectivity_weights
    )

    # Construct the connectivity matrix for the particle network
    connectivity_matrix = define_particle_network_connectivity_matrix(
        n_particles=n_particles, connectivity_bool=connectivity_bool
    )
    
    return connectivity_matrix


def calculate_single_particle_impedance(w, R_ct, C_dl, L_part, D_part, 
                                        D_part_coefficients,
                                        function_type, n_basis,
                                        n_samples, geometry, epsilon=1e-10):
    """
    Calculates the impedance of a single particle using a Randles circuit model
    This function considers the interfacial charge transfer, ion storage in the
    double layer, and diffusion within the particle.

    Parameters:
    ----------
    w : numpy.ndarray
        Array of angular frequencies for which to calculate the impedance.
    R_ct : float
        Charge transfer resistance of the particle.
    C_dl : float
        Double layer capacitance of the particle.
    L_part : float
        Characteristic length of the particle.
    D_part : float
        Diffusion coefficient in the particle.
    D_part_coefficients : numpy.ndarray or None
        Coefficients for spatial variation of D_part described by Legendre 
        polynomials. If None, D_part is considered constant across the particle.
    function_type : str
        Type of function to evaluate spatial variations, only 'legendre' 
        is implemented.
    n_basis : int
        Number of basis functions used for spatial variations.
    n_samples : int
        Number of spatial discretization samples used for constructing the 
        Warburg impedance.
    geometry : str
        The geometry of the particle, affects the calculation of Warburg 
        impedance. Can be 'spherical', 'cylindrical', or 'planar'.

    Returns:
    -------
    numpy.ndarray
        Calculated total impedance of the single particle at the specified 
        frequencies.
    """
    
    # Convert resistive and capacitive components into impedances
    Z_ct = R_ct * np.ones_like(w, dtype=np.cdouble)
    Z_dl = 1 / (1j * w * C_dl)

    # Construct spatial array for evaluating parameters
    x = np.linspace(0, 1, n_samples) * 2 - 1  # Rescale to [-1, 1] for Legendre evaluation

    # Adjust D_part based on spatial variations if coefficients are provided
    y0 = np.log(D_part)
    D_part_evaluated = np.exp(evaluate_parameters(
        x, y0, basis_coefficients=D_part_coefficients, function_type=function_type
    ))

    # Calculate particle's time constant and pseudocapacitance
    t_part = L_part**2 / D_part_evaluated
    C_pc = t_part / R_ct
    Z_parallel = 1 / (1j * np.outer(w, C_pc))

    # Calculate Warburg impedance associated with ion diffusion within particle
    Z_warburg = impedance_transmission_line(
        w, n_samples, Z_series=Z_ct, Z_parallel=Z_parallel, 
        geometry=geometry)

    # Calculate total impedance for the single particle at every frequency
    Z = 1 / (1 / (Z_ct + Z_warburg) + 1 / Z_dl)

    # Bound the determined impedance values between 1/epsilon to epsilon to
    # prevent 0 based division errors in other functions
    phases, magnitudes = np.angle(Z), np.abs(Z)
    bounded_magnitudes = np.where(magnitudes > 1/epsilon, 1/epsilon, magnitudes)
    bounded_magnitudes = np.where(magnitudes < epsilon, epsilon, magnitudes)
    Z = bounded_magnitudes * np.exp(1j * phases)

    # set name of the final output
    Z_single_particle = Z

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

    return Z_single_particle


def pull_particle_network_relevant_numbers(num_particles):
    """
    Calculate relevant numbers for a particle network, including the total
    number of nodes, total possible edges, and the number of edges that can be
    edited.

    Parameters
    ----------
    num_particles : int
        The number of particles in the network.

    Returns
    -------
    tuple
        A tuple containing:
        - num_nodes (int): Total number of nodes in the network.
        - num_total_edges (int): Total number of possible edges (connections) in
          the network.
        - num_editable_edges (int): Number of edges that can be edited based on
          the network configuration.
    """
    num_nodes = 2 + 2 * num_particles  # Including source and drain nodes
    num_total_edges = num_nodes * (num_nodes - 1) / 2
    num_editable_edges = (2 * num_particles + num_particles * (num_particles - 1))

    return num_nodes, num_total_edges, num_editable_edges


def generate_particle_network_admittance_matrix(n_particles, frequency, 
                                                particle_impedances,
                                                resistance_network):
    """
    Generate the admittance matrix for a network of particles connected to each
    other and a backbone structure.

    This function constructs the admittance matrix considering the individual
    particle impedances and the inter-particle resistances. The resulting matrix
    is sparse and constructed for use in simulating the network's electrical
    behavior at a given frequency.

    Parameters
    ----------
    n_particles : int
        Number of particles in the network.
    frequency : float
        Frequency at which the admittance is calculated.
    particle_impedances : numpy.ndarray
        The calculated impedances of individual particles within the network.
        This array should have a length equal to the number of particles.
    resistance_network : numpy.ndarray
        The resistances between particles, as well as between particles and the
        backbone structure. This array should represent all unique particle-
        particle and particle-backbone interactions.

    Returns
    -------
    scipy.sparse.csr_matrix
        The generated sparse complex admittance matrix representing the
        interconnections and individual characteristics of the particle network.
        The shape of the matrix is (n_nodes, n_nodes), where n_nodes includes
        all particle nodes plus source and drain nodes.

    Notes
    -----
    The construction of the admittance matrix accounts for both the inter-
    particle resistances and the inherent impedances of the particles
    themselves. The matrix is formed in COO format for initial construction and
    then converted to CSR format for efficiency in subsequent computations.
    """
    
    # Calculate total number of nodes in the network including source and drain
    n_nodes, _, _ = pull_particle_network_relevant_numbers(n_particles)

    # Admittance will simply be the admittance of the connections in the
    # particle network followed by the admittance of the particle themselves.
    data = np.append(1 / resistance_network, 1 / particle_impedances)
    
    # Define inter-particle connections
    X, Y = np.meshgrid(
        np.arange(1, 1 + n_particles), np.arange(1 + n_particles, n_nodes - 1)
    )
    mask = (Y.flatten() - X.flatten()) != n_particles
    row_interparticle, col_interparticle = X.flatten()[mask], Y.flatten()[mask]
    
    # Define the row and column indices for the raw admittance matrix. Note that
    # the source node will always be the first node, the drain node will always
    # be the last node
    row_indices = np.concatenate([
        np.zeros(n_particles), # source-1st target connections
        np.arange(1 + n_particles, n_nodes - 1), # drain-2nd target connections
        row_interparticle, # interparticle connections
        np.arange(1, 1 + n_particles) # particle impedances
    ]) # particle models
    col_indices = np.concatenate([
        np.arange(1, 1 + n_particles), # source-1st target connections
        np.full(n_particles, n_nodes - 1), # drain-2nd target connections
        col_interparticle, # interparticle connections
        np.arange(1 + n_particles, n_nodes - 1) # particle impedances
    ]) # particle models
    
    # Construct the sparse connectivity matrix, ensuring symmetry
    admittance_matrix = coo_matrix((data, (row_indices, col_indices)),
                                     shape=(n_nodes, n_nodes))
    admittance_matrix = (admittance_matrix + admittance_matrix.T)

    return admittance_matrix.tocsr()


def calculate_particle_network_impedance(w, n_particles,
                                         R_network, connectivity_matrix,
                                         R_ct, C_dl, L_part, D_part,
                                         particle_params, particle_params_coefficients,
                                         inputs=None, epsilon=1e-10):
    """
    Calculates the impedance of a network of particles within a porous
    electrode.

    Evaluates the collective impedance from the inter-particle resistance
    network and the individual particle contributions.

    Parameters
    ----------
    w : numpy.ndarray
        The angular frequencies for which the impedance is calculated.
    n_particles : int
        The total number of particles in the network.
    R_network : numpy.ndarray
        An array containing the resistances between particles and backbone
        structures.
    connectivity_matrix : scipy.sparse.coo_matrix
        A matrix representing the connectivity between particles.
    R_ct, C_dl, L_part, D_part : list or numpy.ndarray
        Lists or arrays of charge transfer resistance, double-layer capacitance, 
        characteristic length, and diffusion coefficient for each particle.
    particle_params : list of dicts
        A list containing dictionaries for each particle's parameters.
    particle_params_coefficients : list of dicts
        A list containing dictionaries for each particle's spatial variation coefficients.
    inputs : dict, optional
        Additional input parameters required for specific calculations, such as 
        functions type and spatial discretization settings.

    Returns
    -------
    Z_network : numpy.ndarray
        The calculated network impedance for each frequency provided in w.

    Notes
    -----
    The impedance calculation accounts for both the resistive and capacitive 
    properties of the particles and their connections, reflecting the complex
    behavior of porous electrode materials.
    """

    # Determine the structure of input data from the number of particles
    _, _, n_variables = pull_particle_network_relevant_numbers(n_particles)

    # Initialize the output impedance array
    Z = np.zeros_like(w, dtype=np.cdouble)

    # Validate the size of the resistance network
    if R_network.size != n_variables:
        raise ValueError(f"Size of R_network does not match the number of "
                         f"variables: {n_particles}")

    # Calculate individual particle impedances
    particle_impedances = [None] * n_particles
    for idx in range(n_particles):
        if particle_params_coefficients[idx]["Particle"]["D_part_coefficients"] is not None:
            if particle_params_coefficients[idx]["Electrode"]["D_part_coefficients"] is not None: # coefficients of coefficients are stored in the input D_part
                D_part_idx = particle_params[idx]["D_part"]
                D_part_coeffs_idx = D_part[idx]
            else: # coefficients only exist at the primary particle level, which are stored int he particle parameters coefficients
                D_part_idx = particle_params[idx]["D_part"]
                D_part_coeffs_idx = particle_params_coefficients[idx]["Particle"]["D_part_coefficients"]
        else: # no spatial variation of the diffusivity at the primary particle level
            D_part_idx = D_part[idx]
            D_part_coeffs_idx = None
        particle_impedances[idx] = calculate_single_particle_impedance(
            w, R_ct[idx], C_dl[idx], L_part[idx], D_part_idx, D_part_coeffs_idx,
            function_type=inputs["Function"]["type"],
            n_basis=inputs["Function"]["n_basis"],
            n_samples=inputs["Particle"]["n_samples"],
            geometry=inputs["Particle"]["geometry"]
        )    
    particle_impedances = np.column_stack(particle_impedances)

    # Calculate network impedance for each frequency
    for idx, frequency in enumerate(w):
        raw_admittance_matrix = generate_particle_network_admittance_matrix(
            n_particles, frequency, particle_impedances[idx, :].flatten(),  
            R_network
        )
        admittance_matrix = raw_admittance_matrix.multiply(connectivity_matrix)
        Z[idx] = calculate_equivalent_impedance(admittance_matrix)
    
    # Bound the determined impedance values between 1/epsilon to epsilon to
    # prevent 0 based division errors in other functions
    phases, magnitudes = np.angle(Z), np.abs(Z)
    bounded_magnitudes = np.where(magnitudes > 1/epsilon, 1/epsilon, magnitudes)
    bounded_magnitudes = np.where(magnitudes < epsilon, epsilon, magnitudes)
    Z = bounded_magnitudes * np.exp(1j * phases)

    # set name of the final output
    Z_network = Z

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

    return Z_network

