import os
import pandas as pd
import numpy as np
import chardet
import configparser
import ast
import shutil
from scipy.interpolate import interp1d

from src.electrode import pull_particle_network_relevant_numbers


def get_file_encoding(file_path):
    """
    Detects and returns the encoding of a given file.
    
    This function opens a file in binary mode and reads its contents to determine the encoding using the `chardet` library. The encoding is then returned as a string.

    Parameters
    ----------
    file_path : str
        The path to the file for which the encoding is to be detected.

    Returns
    -------
    str
        The detected encoding of the file.

    Notes
    -----
    The entire file is read into memory, which may not be efficient for very large files. In such cases, consider reading only a portion of the file to detect its encoding.

    Examples
    --------
    >>> encoding = get_file_encoding("example.txt")
    >>> print(encoding)
    'utf-8'

    Ensure that `chardet` is installed and available in your environment to use this function. If it's not installed, you can install it using `pip install chardet`.
    """
    with open(file_path, 'rb') as file:
        return chardet.detect(file.read())['encoding']


def read_BioLogic_data(file_path, file_encoding=None):
    """
    Reads data from a BioLogic file, using the specified or detected file
    encoding, and assigns column headers based on the last header line.
    
    This function first determines the file encoding either through direct
    specification or automatic detection. It then reads the specified number of
    header lines to identify the number to skip and uses the last header line as
    column names for the DataFrame. It reads the tab-delimited data into a
    pandas DataFrame, handling encoding errors if encountered.
    
    Parameters
    ----------
    file_path : str
        The path to the BioLogic data file to be read.
    file_encoding : str, optional
        The encoding of the file. If None, the encoding is automatically
        detected.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the data read from the file, with appropriate
        headers as column names.
    
    Raises
    ------
    UnicodeDecodeError
        If the specified or detected file encoding is incorrect.
    
    Notes
    -----
    Automatic encoding detection is performed if no encoding is specified. The
    function assumes that the second line of the file contains the number of
    header lines formatted as "Header : X", where X indicates how many lines to
    skip before the data starts, including the line that contains column names.
    
    Examples
    --------
    >>> data = read_BioLogic_data("path/to/your/biologic_file.txt")
    >>> print(data.head())
    
    The output will display the top rows of the DataFrame containing the data
    from the BioLogic file, with the last header line used as column names.
    """
    
    try:
        if file_encoding is None:
            file_encoding = get_file_encoding(file_path)
        
        headers = []
        with open(file_path, 'r', encoding=file_encoding) as file:
            for _ in range(2):  
                # Skip the first line and read the second line for header count
                header_line = file.readline()
            
            # Extract the number of header lines
            header_count = int(header_line.split(':')[1].strip())
            
            # Now, read the headers
            for _ in range(header_count - 2):
                headers.append(file.readline().strip())
        
        # Use the last line read as column names, assuming it contains the 
        # column headers
        column_names = headers[-1].split('\t')
        
        # Adjust skiprows to account for already read lines
        data = pd.read_csv(file_path, delimiter='\t', 
                           skiprows=len(headers)+2,  # Adjusting for the lines 
                           # already read
                           names=column_names, 
                           encoding=file_encoding)

    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(f"Error reading {file_path}: {e}")
    
    return data


def simplify_EIS_dataset(eis_data, num_frequency,
                         frequency_llim=None, frequency_ulim=None):
    """
    Simplifies an Electrochemical Impedance Spectroscopy (EIS) dataset by
    interpolating the measurements to a specified number of log-spaced
    frequencies.

    This function groups the input EIS data by "freq/Hz", averages any duplicate
    frequencies, generates a new set of log-spaced frequencies between the
    minimum and maximum frequency values, and interpolates the other measurement
    values for these new frequencies.

    Parameters
    ----------
    eis_data : pandas.DataFrame
        The input DataFrame containing the EIS dataset. Must include a "freq/Hz"
        column among other measurement columns.
    num_frequency : int
        The number of log-spaced frequency points to generate for the simplified
        dataset.

    Returns
    -------
    pandas.DataFrame
        A simplified DataFrame where "freq/Hz" contains `num_frequency` log-
        spaced frequencies, and other columns contain interpolated values
        corresponding to these frequencies.

    Notes
    -----
    - The function uses linear interpolation for simplifying the dataset. It
      extrapolates values for frequencies outside the original data range.
    - The input DataFrame is expected to have measurement columns that can be
      numerically interpolated. Non-numeric columns are ignored.

    Examples
    --------
    >>> eis_data = pd.DataFrame({
    ...     'freq/Hz': [1, 10, 100, 1000],
    ...     'Zreal': [100, 90, 50, 10],
    ...     'Zimag': [-10, -20, -30, -40]
    ... })
    >>> simplified_data = simplify_EIS_dataset(eis_data, 10)
    >>> print(simplified_data)
    
    The output will show a DataFrame with 10 log-spaced frequency points and
    interpolated 'Zreal' and 'Zimag' values.
    """

    # Group by "freq/Hz" and average duplicates
    average_eis_data = eis_data.groupby('freq/Hz', as_index=False).mean()

    # Generate log-spaced frequencies
    min_freq = average_eis_data['freq/Hz'].min()
    if frequency_llim is not None:
        if min_freq > frequency_llim:
            ValueError("input frequency lower limit is lower than the minimum frequency of the dataset")
        else:
            min_freq = frequency_llim
    max_freq = average_eis_data['freq/Hz'].max()
    if frequency_ulim is not None:
        if max_freq < frequency_ulim:
            ValueError("input frequency upper limit is greater than the maximum frequency of the dataset")
        else:
            min_freq = frequency_llim
    log_spaced_freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), 
                                   num=num_frequency)

    # Interpolate other columns
    interpolated_data = {}
    interpolated_data['freq/Hz'] = log_spaced_freqs
    for column in average_eis_data.columns:
        if column != 'freq/Hz':
            # Skip the first column ("freq/Hz")
            interpolator = interp1d(average_eis_data['freq/Hz'], 
                                    average_eis_data[column], 
                                    bounds_error=False, fill_value="extrapolate")
            interpolated_data[column] = interpolator(log_spaced_freqs)

    simple_eis_data = pd.DataFrame(interpolated_data)

    return simple_eis_data


def read_config(filename):
    """
    Reads a configuration file and returns a dictionary containing its contents.

    Parameters:
        filename (str): The name of the configuration file to read.

    Returns:
        dict: A dictionary representing the contents of the configuration file.
              The dictionary has sections as keys, and each section maps to 
              another dictionary where keys are options and values are 
              corresponding values read from the file.

    Example:
        If the configuration file 'example.cfg' contains:

        [Section1]
        string_variable = Hello World
        integer_variable = 35
        float_variable = 3.14

        [Section2]
        another_string_variable = TestString
        another_integer_variable = 225
        another_float_variable = 2.7661833

        Then calling read_config('example.cfg') would return:

        {
            'Section1': {
                'string_variable': 'Hello World',
                'integer_variable': 35,
                'float_variable': 3.14
            },
            'Section2': {
                'another_string_variable': 'TestString',
                'another_integer_variable': 225,
                'another_float_variable': 2.7661833
            }
        }
    """
     
    config = configparser.ConfigParser()
    config.read(filename)
    
    config_data = {}
    for section in config.sections():
        config_data[section] = {}
        
        for option in config.options(section):
            value = config.get(section, option).strip()  # Strip whitespace

            flag_continue = False
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    flag_continue = True

            if flag_continue:
                # Check for boolean values
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.lower() == 'none':
                    value = None
                else:
                    # Check for list or dictionary values
                    try:
                        # This will convert the string to a list or dict if possible,
                        # or raise ValueError if not.
                        value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):  # SyntaxError covers bad dictionary definitions
                        # If there's an error, leave the value as a string
                        pass

            config_data[section][option] = value
        
    return config_data


def write_config(filename, config_data):
    """
    Writes the configuration data to a file.

    Parameters
    ----------
    filename : str
        The name of the configuration file to write.
    config_data : dict
        A dictionary containing the configuration data to be written. 
        Keys are section names, and values are dictionaries containing options 
        as keys and corresponding values to be written to the file.

    Returns
    -------
    None

    Notes
    -----
    This function writes the configuration data to the specified file using 
    the `ConfigParser` class from the `cfg` module (presumably `cfg` is an alias 
    for `configparser`). It iterates over the keys and values of `config_data` 
    and writes them to the file as sections and options, respectively. 
    The values are converted to strings before writing.

    Example
    -------
    If `config_data` is: 
    {
        'Section1': {
            'string_variable': 'Hello World',
            'integer_variable': 42,
            'float_variable': 3.14
        },
        'Section2': {
            'another_string_variable': 'OpenAI',
            'another_integer_variable': 100,
            'another_float_variable': 2.71828
        }
    }

    Calling `write_config('example.cfg', config_data)` would write the 
    following content to 'example.cfg':

    [Section1]
    string_variable = Hello World
    integer_variable = 42
    float_variable = 3.14

    [Section2]
    another_string_variable = OpenAI
    another_integer_variable = 100
    another_float_variable = 2.71828
    """
    
    config = configparser.ConfigParser()

    for section, options in config_data.items():
        config[section] = {}
        for option, value in options.items():
            config[section][option] = str(value)

    with open(filename, 'w') as configfile:
        config.write(configfile)


def edit_config(filename, section, option, new_value):
    """
    Edits the value of a variable in a configuration file.

    Parameters
    ----------
    filename : str
        The name of the configuration file to edit.
    section : str
        The section containing the variable to be edited.
    option : str
        The name of the variable to be edited.
    new_value : str
        The new value to be assigned to the variable.

    Returns
    -------
    None

    Notes
    -----
    This function uses the `ConfigParser` class from the built-in `configparser`
    module to parse the configuration file and edit the value of the specified 
    variable. If the specified section or option does not exist in the file, 
    this function will create them.

    Example
    -------
    If 'example.cfg' contains:
    
    [Section1]
    string_variable = Hello World
    integer_variable = 42
    float_variable = 3.14
    
    Calling `edit_config('example.cfg', 'Section1', 'string_variable', 
    'New Value')` would edit the 'string_variable' in 'Section1' to 'New Value'.
    """

    config = configparser.ConfigParser()
    config.read(filename)

    # Create the section if it doesn't exist
    if section not in config:
        config[section] = {}

    # Create the option if it doesn't exist
    if option not in config[section]:
        config[section][option] = new_value
    else:
        # Update the value of the existing option
        config[section][option] = new_value

    # Write the changes back to the file
    with open(filename, 'w') as configfile:
        config.write(configfile)


def check_inputs_simulation(inputs):
    """
    Validate the parameters in the 'Simulation' section of the configuration.

    This function checks the 'Simulation' section of a configuration dictionary 
    typically read from .ini files. It ensures that all required fields are
    present  and that their values are of the correct type and within acceptable
    values.

    Parameters
    ----------
    inputs : dict
        A dictionary containing the parameters to be validated. Expected to be
        part of a larger configuration read from a .ini file, specifically the
        'Simulation' section.

    Raises
    ------
    ValueError
        If an unknown field is encountered, or if the values of known fields are
        not of the correct type or within acceptable values.

    Notes
    -----
    The function checks for the following fields: 'fit', 'plot', 'display', 
    and 'save_directory'. 'fit', 'plot', and 'display' should be boolean values.
    'save_directory' should be a string, and if it is 'out', the directory is
    created if it does not exist. Otherwise, 'save_directory' should be an
    existing directory or raise an error.
    """

    valid_fields = ['fit', 'plot', 'display', 'save_directory']

    for field, value in inputs.items():
        if field not in valid_fields:
            raise ValueError(f"Unknown field {field} encountered in the "
                             "simulation inputs.")

        if field in ['fit', 'plot', 'display']:
            if not isinstance(value, bool):
                raise ValueError(f"Specified value for field '{field}' should "
                                 "be boolean.")
        elif field == 'save_directory':
            if value == 'out':
                if not os.path.exists(value):
                    os.makedirs(value)
            else:
                if not os.path.isdir(value):
                    raise ValueError(f"Specified value for field '{field}' "
                    "should be an existing directory.")


def check_inputs_data(inputs):
    """
    Validate the parameters in the 'Data' section of the configuration file.

    This function checks the 'Data' section of a configuration dictionary typically
    read from .ini files. It ensures that all required fields are present and that 
    their values are of the correct type and within acceptable ranges.

    Parameters
    ----------
    inputs : dict
        A dictionary containing the parameters to be validated, expected to be 
        the 'Data' section of a larger configuration read from an .ini file.

    Raises
    ------
    ValueError
        If an unknown field is encountered, or if the values of known fields are
        not of the correct type or within acceptable ranges.

    Notes
    -----
    The function checks for the following fields: 'simplify', 'n_frequency', 
    'frequency_llim', and 'frequency_ulim'. 'simplify' should be a boolean value;
    'n_frequency' should be an integer greater than 1; 'frequency_llim' and 
    'frequency_ulim' should be floating-point numbers representing the lower and 
    upper limits of the frequency range, respectively.
    """

    valid_fields = ['simplify', 'n_frequency', 'frequency_llim', 'frequency_ulim']

    for field, value in inputs.items():
        if field not in valid_fields:
            raise ValueError(f"Unknown field {field} encountered in the "
                             "data inputs.")

        if field in ['simplify']:
            if not isinstance(value, bool):
                raise ValueError(f"Field '{field}' should be boolean.")

        elif field in ['n_frequency']:
            if not (isinstance(value, int) and value > 0):
                raise ValueError(f"Field '{field}' should be a positive integer.")

        elif field in ['frequency_llim', 'frequency_ulim']:
            if not (isinstance(value, float) and value > 0):
                raise ValueError(f"Field '{field}' should be a positive float.")


def check_inputs_function(inputs):
    """
    Validate parameters specified in the 'Function' section of a configuration.

    This function ensures that the parameters provided for mathematical
    functions, such as basis functions for approximation, are correctly
    formatted and fall within acceptable ranges. It specifically checks for the
    'type' of function and the number of basis functions ('n_basis') used in the
    model.

    Parameters
    ----------
    inputs : dict
        A dictionary containing the function-related parameters to be validated.
        Expected keys are 'type' and 'n_basis'.

    Raises
    ------
    ValueError
        If an unknown field is encountered, or if the values of known fields are
        not of the correct type or within acceptable ranges.

    Notes
    -----
    Currently, only 'legendre' type functions are implemented. The number of
    basis functions, 'n_basis', must be an integer greater than 1.

    Examples
    --------
    >>> inputs_function = {'type': 'legendre', 'n_basis': 5}
    >>> check_inputs_function(inputs_function)
    Validates that the function type is 'legendre' and that there are five basis
    functions specified.
    """
    
    valid_fields = ['type', 'n_basis']

    for field, value in inputs.items():
        if field not in valid_fields:
            raise ValueError(f"Unknown field '{field}' encountered in the "
                             "function inputs.")
        
        if field == 'type':
            if not (isinstance(value, str) and value in ['legendre']):
                raise ValueError(f"Field '{field}' should be 'legendre'.")
        
        elif field == 'n_basis':
            if not (isinstance(value, int) and value > 0):
                raise ValueError(f"Field '{field}' should be a postiive integer.")


def check_inputs_plotting(inputs):
    """
    Validate the parameters in the 'Plotting' section of the configuration.

    This function checks the 'Plotting' section of a configuration dictionary 
    typically read from .ini files. It ensures that all required fields are
    present  and that their values are of the correct type and within acceptable
    values.
    """

    valid_fields = ['type', 'directory']

    for field, value in inputs.items():
        if field not in valid_fields:
            raise ValueError(f"Unknown field {field} encountered in the "
                             "plotting inputs.")

        if field in ['type']:
            if not isinstance(value, str):
                raise ValueError(f"Specified value for field '{field}' should "
                                 "be string.")
            split_values = value.split(",")
            for split_value in split_values:
                if split_value not in ['Nyquist', 'Bode-magnitude', 'Bode-phase']:
                    raise ValueError(f"Specified plotting type '{split_value}' "
                                     "is not an implemented plot type")
        elif field in ['directory']:
            if not isinstance(value, str):
                raise ValueError(f"Specified value for field '{field}' should "
                                 "be string.")


def check_inputs_optimization(inputs):
    """
    Validates the parameters within the 'Optimization' section of a
    configuration file.

    Ensures that all necessary fields for conducting optimization, such as
    method and parameters specific to the optimization process, are present and
    correctly formatted.

    Parameters
    ----------
    inputs : dict
        A dictionary containing optimization-related parameters, typically
        including keys such as 'method', 'temperature', 'chain_length',
        'display_frequency', 'display_value', 'plot_frequency', and
        'save_frequency', specifying the optimization method, parameters for the
        optimization process, and settings related to the display and saving of
        optimization progress and results.

    Raises
    ------
    ValueError
        If an unknown configuration field is encountered, or if the values of
        known fields are not of the expected type or within acceptable ranges.

    Examples
    --------
    >>> inputs = {
        'method': 'mcmc',
        'temperature': 300.0,
        'chain_length': 10000,
        'display_frequency': 100,
        'display_value': 'best',
        'plot_frequency': 500,
        'save_frequency': 1000
    }
    >>> check_inputs_optimization(inputs)
    Validates all provided 'Optimization' parameters, ensuring correctness and acceptable ranges.
    """

    valid_fields = ['method', 'burn_length', 'burn_temperature', 
                    'temperature', 'chain_length',
                    'display_frequency', 'display_value', 'plot_frequency',
                    'save_frequency']

    for field, value in inputs.items():
        if field not in valid_fields:
            raise ValueError(f"Unknown field '{field}' encountered in the "
                             "optimization inputs.")

        if field == 'method':
            if not (isinstance(value, str) and value == 'mcmc'):
                raise ValueError("Currently, only the 'mcmc' optimization "
                                 "method is implemented.")
        
        elif field == 'display_value':
            if not (isinstance(value, str) and value in ['best', 'current']):
                raise ValueError("'display_value' should be either 'best' or "
                                 "'current'.")
        
        elif field.endswith('_frequency') or field.endswith('_length'):
            if not (isinstance(value, int) and value > 0):
                raise ValueError(f"Field '{field}' should be a positive integer.")
        
        elif field.endswith('temperature'):
            if not (isinstance(value, float) and value > 0):
                raise ValueError("Field 'temperature' should be a positive float.")
        

def check_inputs_objective(inputs):
    """
    Validates the parameters within the 'Objective' configuration section.

    This function ensures that all necessary fields required for defining the
    properties of the objective function calculation are present and valid. It
    verifies the correctness of the types and acceptable values for each
    configuration parameter.

    Parameters
    ----------
    inputs : dict
        A dictionary containing the parameters related to the objective function
        configuration.

    Raises
    ------
    ValueError
        If an unknown configuration field is encountered or if any of the
        parameters are of incorrect type or have unacceptable values.

    Notes
    -----
    The expected fields include various settings for misfit, parameter
    complexity weights, temperature bounds for different processes, and settings
    for the spatial basis scale and function weights.
    """
    
    # Define valid fields for the objective configurations
    valid_fields = [
        'weights', 'misfit_setting', 'parameter_complexity_weights',
        't_lyte_min', 't_lyte_max', 't_lyte_bounds_temperature',
        't_rxn_min', 't_rxn_max', 't_rxn_bounds_temperature',
        't_diff_min', 't_diff_max', 't_diff_bounds_temperature',
        'r_network_min', 'r_network_max', 'r_network_bounds_temperature',
        'r_ct_min', 'r_ct_max', 'r_ct_bounds_temperature',
        'r_lyte_min', 'r_lyte_max', 'r_lyte_bounds_temperature',
        'l_part_min', 'l_part_max', 'l_part_bounds_temperature',
        'connectivity_bounds_temperature',
        'basis_function_elastic_weights', 'spatial_basis_scale',
    ]

    # Check and validate each field provided in the inputs
    for field, value in inputs.items():
        if field not in valid_fields:
            raise ValueError(f"Unknown field '{field}' encountered in "
                             "objective inputs.")

        # Check fields for temperature bounds and scales
        if (field.endswith('_min') or field.endswith('_max') 
            or field.endswith('_bounds_temperature')
            or field == 'spatial_basis_scale'):
            if not (isinstance(value, float) and value > 0):
                raise ValueError(f"Field '{field}' should be a positive float.")

        # Check fields for weight configurations
        elif field.endswith('_weights'):
            if not (isinstance(value, list) and np.isclose(sum(value), 1)):
                raise ValueError(f"Field '{field}' should be a list of values "
                                 "that sum to 1.")

        # Check fields for misfit settings
        elif field == 'misfit_setting':
            if value not in ['ReOnly', 'ImOnly', 'ReIm']:
                raise ValueError("Field 'misfit_setting' must be one of "
                                 "['ReOnly', 'ImOnly', 'ReIm'].")


def check_inputs_electrode(inputs, n_particles):
    """
    Validate the parameters in the 'Electrode' section of the configuration.

    Ensures that all fields required for defining electrode properties are
    present and valid by checking for correct types and acceptable values
    for each parameter related to the electrode configuration.

    Parameters
    ----------
    inputs : dict
        Dictionary containing parameters related to the electrode, 
        typically obtained from the 'Electrode' section of a configuration file.
    n_particles : int
        Number of particles defined in the electrode configuration.

    Raises
    ------
    ValueError
        Raised if an unknown field is encountered, if the number of particles
        is not an integer or is negative, or if any parameter is of an incorrect
        type or holds unacceptable values.

    Notes
    -----
    The expected fields in the input dictionary include resistances (bulk, 
    ion in electrolyte, electron in electrolyte), time constant for electrolyte
    transport, number of samples, and various parameters related to particle
    characteristics such as variations and coefficients for different
    resistances and lengths.
    """

    # Validate fields are known and correctly formatted
    valid_fields = [
        'geometry', 'n_samples',
        'r_bulk', 'r_electrolyte_ion', 'r_electrolyte_electron',
        'c_electrolyte', 
        'r_ct_variations', 'r_ct_coefficients',
        'c_dl_variations', 'c_dl_coefficients',
        'l_part_variations', 'l_part_coefficients',
        'd_part_variations', 'd_part_coefficients',
        'r_source_target_variations', 'r_source_target_coefficients',
        'r_drain_target_variations', 'r_drain_target_coefficients',
        'r_target_target_variations', 'r_target_target_coefficients'
    ]

    # Iterate through each provided field and validate
    for field, value in inputs.items():
        if field not in valid_fields:
            raise ValueError(f"Unknown field '{field}' encountered in the "
                             "electrode inputs.")

        # Validate resistances and time constants
        if field in ['r_bulk', 'r_electrolyte_ion', 'r_electrolyte_electron',
                     'c_electrolyte']:
            if not (isinstance(value, float) and value > 0):
                raise ValueError(f"Field '{field}' should be a positive float.")

        # Validate sample count
        elif field == 'n_samples':
            if not (isinstance(value, int) and value > 0):
                raise ValueError(f"Field '{field}' should be a positive integer.")

        # Validate boolean variations
        elif field.endswith('_variations'):
            if not isinstance(value, bool):
                raise ValueError(f"Field '{field}' should be a boolean.")

        # Validate coefficient arrays
        elif field.endswith('_coefficients'):
            if 'target_target' in field:
                expected_length = n_particles * (n_particles - 1)
            else:
                expected_length = n_particles
            if not (isinstance(value, list) and len(value) == expected_length):
                raise ValueError(f"Field '{field}' should be a list of length "
                                 f"{expected_length}.")
        
        elif field == 'geometry':
            if not (isinstance(value, str) and value in ['planar']):
                raise ValueError("'geometry' must be planar for the electrode.")


def check_inputs_particle_network(inputs):
    """
    Validate parameters in the 'Particle Network' section of a configuration.

    This function checks all necessary fields for defining properties of the
    particle network to ensure they are present and hold valid values. It
    verifies correct data types and acceptable ranges for each parameter related
    to the particle network settings in a simulation configuration.

    Parameters
    ----------
    inputs : dict
        A dictionary containing the parameters for the particle network
        settings, typically obtained from the 'Particle Network' section of a
        configuration file.

    n_particles : int
        The specified number of particles within the network configuration,
        which determines the necessary size and structure of the network
        parameters.

    Raises
    ------
    ValueError
        Raises an error if an unknown field is encountered, if any parameter is
        of an incorrect type, if list parameters do not have the expected
        length, or if numerical values fall outside of their permissible ranges.

    Notes
    -----
    This function specifically validates fields including 'n_particles',
    'connectivity_*' (source_target, drain_target, target_target), and
    'r_*' (source_target, drain_target, target_target) ensuring each is
    correctly formatted and logically consistent with the expected network
    topology.
    """
    
    n_particles = inputs["n_particles"]
    if not isinstance(n_particles, int) or n_particles < 0:
        raise ValueError("The 'n_particles' value should be a non-negative "
                         "integer.")

    valid_fields = ['n_particles',
                    'connectivity_source_target', 'connectivity_drain_target',
                    'connectivity_target_target', 'r_source_target',
                    'r_drain_target', 'r_target_target']

    for field, value in inputs.items():
        if field not in valid_fields:
            raise ValueError(f"Unknown field '{field}' found in particle "
                             "network configuration.")

        if field.startswith('connectivity_'):
            if 'target_target' in field:
                expected_length = n_particles * (n_particles - 1)
            else:
                expected_length = n_particles
            if not (isinstance(value, list)
                    and len(value) == expected_length
                    and all(isinstance(v, float) and (0 <= v <= 1) for v in value)):
                raise ValueError(f"Field '{field}' must be a list of floats "
                                 f"between 0 and 1 of length {expected_length}.")

        elif field.startswith('r_'):
            if 'target_target' in field:
                expected_length = n_particles * (n_particles - 1)
            else:
                expected_length = n_particles
            if not (isinstance(value, list)
                    and len(value) == expected_length
                    and all(isinstance(v, float) and v > 0 for v in value)):
                raise ValueError(f"Field '{field}' must be a list of positive "
                                 f"floats of length {expected_length}.")


def check_inputs_particle(inputs, n_particles):
    """
    Validate the parameters in the 'Particle' section of the configuration.

    This function reviews the 'Particle' section from a configuration dictionary,
    ensuring all required fields are present, correctly typed, and within
    acceptable bounds. It is typically used for configurations derived from
    .ini files.

    Parameters
    ----------
    inputs : dict
        Dictionary of parameters related to particle properties.
    n_particles : int
        Number of particles described in the particle network.

    Raises
    ------
    ValueError
        Raised if an unknown field is encountered, if any parameter values are 
        of the incorrect type, or if list parameters do not have the expected
        length.

    Notes
    -----
    Expected fields include 'model', 'geometry', 'R_ct', 'C_dl', 'L_part',
    'D_part', 'n_samples', 'D_part_variations', and 'D_part_coefficients'.
    'model' and 'geometry' should be strings, 'R_ct', 'C_dl', 'L_part', and
    'D_part' should be lists of floats with length equal to `n_particles`,
    'n_samples' should be a positive integer, and 'D_part_variations' should be
    a boolean. 'D_part_coefficients' should be a list of coefficients
    corresponding to each particle.
    """
    
    valid_fields = ['model', 'geometry', 'r_ct', 'c_dl', 'l_part', 'd_part',
                    'n_samples', 'd_part_variations', 'd_part_coefficients']

    for field, value in inputs.items():
        if field not in valid_fields:
            raise ValueError(f"Unknown field '{field}' encountered in particle "
                             "inputs.")

        if field in ['model', 'geometry']:
            if not isinstance(value, str):
                raise ValueError(f"Field '{field}' should be a string.")
            if field == 'model' and value not in ['Randles']:
                raise ValueError("Currently, only 'Randles' circuit models are "
                                 "implemented.")
            if (field == 'geometry'
                and value not in ['planar', 'cylindrical', 'spherical']):
                raise ValueError(f"Geometry '{value}' is not implemented.")

        elif field in ['r_ct', 'c_dl', 'l_part', 'd_part']:
            if not (isinstance(value, list)
                    and len(value) == n_particles
                    and all(isinstance(v, float) and v > 0 for v in value)):
                raise ValueError(f"Field '{field}' should be a list of positive"
                                 " floats with length equal to the number of "
                                 "particles.")

        elif field == 'n_samples':
            if not (isinstance(value, int) and value > 0):
                raise ValueError(f"Field '{field}' needs to be a positive "
                                 "integer.")

        elif field.endswith('coefficients'):
            if not (isinstance(value, list) and len(value) == n_particles):
                raise ValueError(f"Field '{field}' needs to be a list of "
                                 "length equal to the number of particles.")

        elif field.endswith('variations'):
            if not isinstance(value, bool):
                raise ValueError(f"Field '{field}' needs to be a boolean.")


def validate_configuration(inputs):
    """
    Validate configuration parameters from an input dictionary typically read
    from .ini files.

    This function checks each section of the configuration dictionary, ensuring
    all required fields are present and correctly formatted. It delegates
    validation to more specific functions based on the section of the
    configuration.

    Parameters
    ----------
    inputs : dict
        A dictionary containing configuration parameters, categorized under keys
        such as 'Simulation', 'Data', 'Function', 'Plotting', 'Fitting',
        'Electrode', 'Particle Network', and 'Particle'.

    Raises
    ------
    ValueError
        If an unknown field is encountered in the inputs dictionary, or if
        specific checks within individual sections fail.

    Examples
    --------
    >>> inputs = {
            'Simulation': {'time_step': 0.01, 'duration': 10},
            'Data': {'path': 'data.csv', 'delimiter': ','}
        }
    >>> check_inputs(inputs)
    This will validate the 'time_step' and 'duration' fields under 'Simulation'
    and the 'path' and 'delimiter' fields under 'Data'.

    Notes
    -----
    The function iterates through the provided categories in the input
    dictionary and calls specific checking functions for each section. This
    approach ensures that each section of the configuration is validated
    according to its unique requirements.
    """

    valid_fields = ['Simulation', 'Data', 'Function', 'Plotting',
                    'Optimization', 'Objective',
                    'Electrode', 'Particle Network', 'Particle']

    for field, values in inputs.items():
        if field not in valid_fields:
            raise ValueError(f"Unknown field '{field}' encountered in the "
                             "inputs dictionary.")
        
        if field == 'Simulation':
            check_inputs_simulation(values)
        elif field == 'Data':
            check_inputs_data(values)
        elif field == 'Function':
            check_inputs_function(values)
        elif field == 'Plotting':
            check_inputs_plotting(values)
        elif field == 'Optimization':
            check_inputs_optimization(values)
        elif field == 'Objective':
            check_inputs_objective(values)
        elif field == 'Electrode':
            check_inputs_electrode(values,
                                   inputs["Particle Network"]["n_particles"])
        elif field == 'Particle Network':
            check_inputs_particle_network(values)
        elif field == 'Particle':
            check_inputs_particle(values, 
                                  inputs["Particle Network"]["n_particles"])


def extract_spatial_variations(inputs):
    """
    Extracts entries from a nested dictionary where keys end with '_variations',
    removing this suffix from the keys in the output dictionary.

    Parameters
    ----------
    inputs : dict
        The original nested dictionary containing configuration parameters,
        including those ending with '_variations'.

    Returns
    -------
    dict
        A new nested dictionary containing only the keys that ended with
        '_variations' in the original dictionary, but with this suffix removed
        from the keys.

    Examples
    --------
    >>> inputs = {
            'Electrode': {'r_bulk_variations': True, 'other_param': 5},
            'Particle': {'D_variations': False, 'geometry': 'spherical'}
        }
    >>> extract_variations(inputs)
    {'Electrode': {'r_bulk': True}, 'Particle': {'D': False}}
    """
    
    new_dict = {}
    for section, params in inputs.items():
        # Initialize the sub-dictionary for this section, if there are variations
        section_dict = {}
        for key, value in params.items():
            # Check if this key represents a variation and is a boolean
            if key.endswith('_variations') and isinstance(value, bool):
                # Remove '_variations' from the key
                new_key = key.replace('_variations', '')
                section_dict[new_key] = value
        if section_dict:  # Add this section to the new dictionary if it's not empty
            new_dict[section] = section_dict
    
    return new_dict


def boolean_dict_to_list_spatial_variations(spatial_variation_flag_dictionary,
                                            n_particles, n_basis):
    """

    Generates a list of boolean values indicating whether there are spatial
    variations for each model parameter across multiple particles.

    This function creates a boolean list that reflects the spatial variation
    status (True if varies spatially, False otherwise) for parameters of
    multiple particles based on the input boolean dictionary.

    Parameters
    ----------
    n_particles : int
        The number of particles being simulated in the model.
    boolean_dict : dict
        A dictionary where keys represent the names of parameters and values
        indicate whether the parameter varies spatially (True) or not (False).
    legendre_basis : int
        The number of Legendre polynomial coefficients used to describe spatial
        variations of parameters.

    Returns
    -------
    spatial_variations : list of bool
        A list where each element indicates whether a parameter associated with
        a particle or network varies spatially. The length of this list is
        determined by the number of particles, the number of parameters per
        particle, and the number of network parameters.

    Notes
    -----
    The spatial variations list is structured as follows: particle-specific
    parameters are listed first for all particles, followed by network
    parameters. If a parameter is indicated to have spatial variations,
    additional boolean values are added to represent each Legendre basis
    coefficient.

    Examples
    --------
    >>> n_particles = 3
    >>> boolean_dict = {'lnD_part_coefficients': True, 'resistance_network':
    False, 'lnR_ct': True}
    >>> legendre_basis = 4
    >>> generate_boolean_spatial_variations(n_particles, boolean_dict,
    legendre_basis)
    [True, True, True, False, False, False, True, True, True, True, True,
    True, True, True, True]
    """
    
    _, _, n_variables = pull_particle_network_relevant_numbers(n_particles)
    
    spatial_variations = []

    # Check for spatial variations in particle parameters
    spatial_variations.extend(
        [spatial_variation_flag_dictionary.get('lnD_part', False)] * n_particles)

    # Check for spatial variations in network parameters
    spatial_variations.extend(
        [spatial_variation_flag_dictionary.get('resistance_network', False)] * n_variables)

    # Check for spatial variations in each particle's specific parameters
    for _ in range(n_particles):
        for field in ['lnR_ct', 'lnC_dl', 'lnL_part']:
            spatial_variations.append(spatial_variation_flag_dictionary.get(field, False))
        # Extend spatial variations for Legendre coefficients if applicable
        spatial_variations.extend(
            [spatial_variation_flag_dictionary.get('lnD_part_coefficients', False)] * n_basis)
    
    return spatial_variations


def pack_inputs(inputs):
    """
    Packs the inputs dictionary into a list of parameters (x), a list of
    characteristic step size (del_x), the dictionary of boolean values
    indicating if there are spatial variations (`boolean_dict`), and the vector
    of the spatial variations that exist in the system (`spatial_variations`). 
    """
    
    # Create the boolean dictionary and boolean list for identifying variables
    # that have variations over space.
    bool_spatial_variations = extract_spatial_variations(inputs)
    n_basis = inputs["Function"]["n_basis"]
    n_particles = inputs["Particle Network"]["n_particles"]
    _, _, n_variables = pull_particle_network_relevant_numbers(n_particles)
    
    # Highest level
    x = np.log([inputs["Electrode"]["r_bulk"]])
    delx = np.array([0.05])

    # Series element in Electrode scale
    x = np.append(x, np.log([inputs["Electrode"]["r_electrolyte_ion"],
                             inputs["Electrode"]["r_electrolyte_electron"]]))
    delx = np.append(delx, np.full(2, 0.10))

    
    # Parallel element in Electrode scale. This is just another circuit, so we
    # start with the basic circuit element
    x = np.append(x, np.log(inputs["Electrode"]["c_electrolyte"]))
    delx = np.append(delx, np.full(1, 0.10))
    
    # The basic circuit element is in parallel with a particle network
    # Connection probabilities and network parameters
    connectivity_particle_network = np.concatenate((
        inputs["Particle Network"]["connectivity_source_target"],
        inputs["Particle Network"]["connectivity_drain_target"],
        inputs["Particle Network"]["connectivity_target_target"]))
    ln_R_particle_network = np.log(np.concatenate((
        inputs["Particle Network"]["r_source_target"],
        inputs["Particle Network"]["r_drain_target"],
        inputs["Particle Network"]["r_target_target"])))
    x = np.append(
        x, np.concatenate((connectivity_particle_network, ln_R_particle_network))
    )
    delx = np.append(
        delx, np.concatenate((np.full(n_variables, 0.025), np.full(n_variables, 0.10)))
    )

    # Variations of the particle network parameters
    fields = ['r_source_target', 'r_drain_target', 'r_target_target']
    for field in fields:
        if bool_spatial_variations["Electrode"][field]:
            for coeffs in inputs["Electrode"][field+'_coefficients']:
                x = np.append(x, coeffs)
                delx = np.append(delx, np.full(n_basis, 0.1))
    
    # Particle Scale
    for idx in range(n_particles):
        x_particle = np.log([inputs["Particle"]["r_ct"][idx],
                            inputs["Particle"]["c_dl"][idx],
                            inputs["Particle"]["l_part"][idx],
                            inputs["Particle"]["d_part"][idx],])
        x = np.append(x, x_particle)
        delx = np.append(delx, np.full(4, 0.10))
        
        # variations on the particle scale
        if bool_spatial_variations["Particle"]["d_part"]:
            coeffs = inputs["Particle"]["d_part_coefficients"][idx]
            x = np.append(x, coeffs)
            delx = np.append(delx, np.full(n_basis, 0.1))
    
        # variations on the electrode scale
        fields = ['r_ct', 'c_dl', 'l_part']
        for field in fields:
            if bool_spatial_variations["Electrode"][field]:
                coeffs = inputs["Electrode"][field+'_coefficients']
                x = np.append(x, coeffs)
                delx = np.append(delx, np.full(n_basis, 0.1))
        fields = ['d_part']
        for field in fields:
            if bool_spatial_variations["Electrode"][field]:
                coeffs = inputs["Electrode"][field+'_coefficients']
                if bool_spatial_variations["Particle"][field]:
                    for coeff in coeffs:
                        x = np.append(x, coeff)
                        delx = np.append(delx, np.full(n_basis, 0.1))
                else:
                    x = np.append(x, coeffs)
                    delx = np.append(delx, np.full(n_basis, 0.1))

    return x, delx, bool_spatial_variations


