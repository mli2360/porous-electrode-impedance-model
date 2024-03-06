import numpy as np


def impedance_transmission_line(frequency, n_samples, Z_series, Z_parallel,
                                boundary_condition='sp', geometry='planar'):
    """
    Calculates the impedance of a transmission line segmented into different 
    samples, applying specified boundary conditions.

    Parameters:
    ----------
    frequency : numpy.ndarray
        An array of frequencies at which the line's impedance is calculated.
    n_samples : int
        The number of segments along the transmission line.
    Z_series : numpy.ndarray
        A 1D or 2D array containing the series impedance values for each
        segment. If 1D, it should match the length of `frequency`; if 2D, it
        should be of size (frequency length, n_samples).
    Z_parallel : numpy.ndarray
        A 1D or 2D array containing the parallel impedance values for each
        segment. The size requirements are the same as `Z_series`.
    boundary_condition : str, optional
        The type of boundary condition at the transmission line's end. Options 
        are 'sp' (series-parallel), 's' (series), or 'p' (parallel), with 'sp' 
        being the default.
    geometry : str, optional
        The geometry of the transmission line which can be 'planar',
        'cylindrical', or 'spherical'. The default geometry is 'planar'.

    Returns:
    -------
    numpy.ndarray
        The impedance at each frequency for the entire transmission line.

    Raises:
    ------
    ValueError
        If the shapes of `Z_series` or `Z_parallel` do not match the expected 
        dimensions based on `frequency` and `n_samples`.
        the transmission line at different frequencies.
    """

    # Validate the dimensions of Z_series
    if Z_series.ndim == 1:
        if Z_series.shape != frequency.shape:
            raise ValueError("Z_series must have the same length as frequency.")
    elif Z_series.ndim == 2:
        if not (Z_series.shape == (len(frequency), n_samples)):
            raise ValueError("2D Z_series must have dimensions (frequency "
                             "length, n_samples).")

    # Validate the dimensions of Z_parallel
    if Z_parallel.ndim == 1:
        if Z_parallel.shape != frequency.shape:
            raise ValueError("Z_parallel must have the same length as frequency.")
    elif Z_parallel.ndim == 2:
        if not (Z_parallel.shape == (len(frequency), n_samples)):
            raise ValueError("2D Z_parallel must have dimensions (frequency "
                             "length, n_samples).")
        
    # Determine geometric elements based on the specified geometry
    positions, areas, volumes = define_geometric_elements_mesh(
        geometry=geometry, discretizations=n_samples
    )

    # Construct transmission line by looping through each segment's addition.
    for i in range(n_samples):
        # Define the sample for the segment
        Z_series_sample = Z_series[:, i] if Z_series.ndim == 2 else Z_series
        Z_parallel_sample = Z_parallel[:,i] if Z_parallel.ndim == 2 else Z_parallel
        
        # Define the infinitesimal element for the segment to be added
        Z_series_inf = \
            construct_infinitesimal_element(Z_series_sample, 'series',
                                            1/n_samples, areas[i])
        Z_parallel_inf = \
            construct_infinitesimal_element(Z_parallel_sample, 'parallel',
                                            1/n_samples, areas[i])

        # Apply boundary condition at the start of the line
        if i == 0:
            if boundary_condition == 'sp':
                Z = Z_parallel_inf + Z_series_inf
            elif boundary_condition == 's':
                Z = Z_series_inf
            elif boundary_condition == 'p':
                Z = Z_parallel_inf
            else:
                raise ValueError("Boundary condition must be 'sp', 'p', or 's'.")
        else:
            # Accumulate impedance for intermediate sections
            Z = 1 / (Adm + 1 / Z_parallel_inf) + Z_series_inf

        Adm = 1 / Z  # Update admittance for next section
    
    return Z
    

def solve_telegraph_equations(frequency, Z_series, Z_parallel, n_samples,
                            boundary_condition='sp', geometry='planar'):
    """
    Calculates the impedance for an electrochemical system modeled as a
    transmission line using the telegrapher's equations.

    Parameters
    ----------
    frequency : numpy.ndarray
        The frequencies at which to calculate impedance.
    Z_series : numpy.ndarray
        The series impedance component.
    Z_parallel : numpy.ndarray
        The parallel impedance component.
    n_samples : int
        The number of samples or segments in the transmission line model.
    boundary_condition : str, optional
        The boundary condition applied at the transmission line's end.
        Options are 'sp' (series-parallel), 's' (series), and 'p' (parallel).
        The default is 'sp'.
    geometry : str, optional
        The geometry of the transmission line. Options are 'planar',
        'cylindrical', and 'spherical'. The default is 'planar'.

    Returns
    -------
    Z : numpy.ndarray
        The calculated impedance at each frequency.

    Raises
    ------
    ValueError
        If an unsupported boundary condition is specified.
    """
    
    positions, areas, volumes = define_geometric_elements_mesh(
        geometry=geometry, discretizations=n_samples
    )
    
    Z = np.full_like(frequency, np.inf, dtype=np.cdouble)
    Adm = 1 / Z
    
    Z_series_0 = Z_series / n_samples / areas[0]
    Z_parallel_0 = Z_parallel / n_samples / areas[0]

    if boundary_condition not in ['sp', 'p', 's']:
        raise ValueError("Boundary condition must be one of ['sp', 'p', 's'].")

    # Apply boundary condition
    if boundary_condition == 'sp':
        Z = 1 / (Adm + 1 / Z_parallel_0) + Z_series_0
    elif boundary_condition == 's':
        Z = Z_series_0
    elif boundary_condition == 'p':
        Z = 1 / (Adm + 1 / Z_parallel_0)

    Adm = 1 / Z
        
    for i in range(1, n_samples):
        Z_series_i = Z_series / n_samples / areas[i]
        Z_parallel_i = Z_parallel * n_samples / areas[i]
        Z = 1 / (Adm + 1 / Z_parallel_i) + Z_series_i

        Adm = 1 / Z

    return Z


def define_geometric_elements_mesh(geometry, discretizations):
    """
    Defines normalized geometric elements for different geometries.

    Parameters
    ----------
    geometry : str
        The geometry of the system. Options are 'planar', 'cylindrical', and
        'spherical'.
    discretizations : int
        The number of segments in the transmission line model.

    Returns
    -------
    positions : numpy.ndarray
        The normalized positions of each segment.
    areas : numpy.ndarray
        The normalized areas of each segment, adjusted for geometry.
    volumes : numpy.ndarray
        The normalized volumes of each segment, adjusted for geometry.

    Raises
    ------
    ValueError
        If an unsupported geometry is specified.
    """
    
    if geometry not in ['planar', 'cylindrical', 'spherical']:
        raise ValueError(f"Geometry '{geometry}' is not supported. Choose from "
                         + "'planar', 'cylindrical', 'spherical'.")

    if geometry == "planar":
        area_coeff, area_exponent, volume_coeff, volume_exponent = 1, 0, 1, 1
    elif geometry == 'cylindrical':
        area_coeff, area_exponent, volume_coeff, volume_exponent = 1, 1, 1, 2
    elif geometry == 'spherical':
        area_coeff, area_exponent, volume_coeff, volume_exponent = 1, 2, 1, 3
    
    positions = np.linspace(0 + 1/(2*discretizations), 1 - 1/(2*discretizations),
                            discretizations)
    areas = area_coeff * positions**area_exponent
    
    nodes = np.linspace(0, 1, discretizations+1)
    volumes = np.diff(volume_coeff * nodes**volume_exponent)

    return positions, areas, volumes


def construct_infinitesimal_element(Z_element, connectivity, discretization_size, differential_factor):
    """
    Calculates the impedance of a near-infinitesimal element of an
    electrochemical transmission line model.

    This function computes the impedance of a differential element based on the
    total impedance value provided for the element, taking into account the
    discretization size, the differential factor which is influenced by the
    geometry of the system, and whether the connectivity is in series or
    parallel.

    Parameters
    ----------
    Z_element : float or numpy.ndarray
        The total bulk impedance value(s) at various frequencies. Can be a
        scalar or an array of impedance values corresponding to different
        frequencies.
    connectivity : {'series', 'parallel'}
        Specifies how the infinitesimal element is connected in the circuit.
        Must be either 'series' or 'parallel'.
    discretization_size : float
        The size of the discretization step used in the differential
        approximation. Represents how finely the transmission line is divided
        into differential elements.
    differential_factor : float
        A factor that accounts for the geometry of the transmission line or
        electrochemical cell. This factor adjusts the impedance calculation to
        reflect the specific geometric considerations of the model.

    Returns
    -------
    Z_infinitesimal : float or numpy.ndarray
        The calculated impedance of the infinitesimal element. The type (scalar
        or array) will match that of `Z_element`.

    Raises
    ------
    ValueError
        If `connectivity` is not one of the supported options ('series' or\
        'parallel').

    Examples
    --------
    >>> Z_bulk = 100  # Ohms
    >>> construct_infinitesimal_element(Z_bulk, 'series', 0.01, 1)
    1.0

    >>> construct_infinitesimal_element(Z_bulk, 'parallel', 0.01, 1)
    10000.0
    """
    
    if connectivity not in ['series', 'parallel']:
        raise ValueError(f"Connectivity '{connectivity}' is not supported. "
                         + f"Choose from 'series' or 'parallel'.") 
    
    if connectivity == 'series':
        Z_infinitesimal = Z_element * (discretization_size / differential_factor)
    else:  # connectivity == 'parallel'
        Z_infinitesimal = Z_element / (discretization_size * differential_factor)
    
    return Z_infinitesimal


