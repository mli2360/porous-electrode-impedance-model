import os
import matplotlib.pyplot as plt
from datetime import datetime

from src.data_io import (read_BioLogic_data,
                         simplify_EIS_dataset,
                         read_config,
                         validate_configuration,
                         pack_inputs)
from src.objective import calculate_objective_function
from src.plot import create_impedance_plot
from src.optimization import metropolis_hastings_mcmc


def main(config_file, data_file):
    """
    Main execution function for running simulations based on configuration and
    data files.

    Parameters
    ----------
    config_file : str
        Path to the configuration file.
    data_file : str
        Path to the data file containing the Electrochemical Impedance
        Spectroscopy data.

    Returns
    -------
    None
        The function does not return any value but performs the simulation,
        optimization, and plotting based on the inputs.
    """

    # Read in configuration file
    inputs = read_config(config_file)
    validate_configuration(inputs)

    # Create new directory for saving simulation results
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(inputs["Simulation"]["save_directory"], current_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    inputs["Simulation"]["save_directory"] = save_dir

    # Prepare input parameters
    x0, delx, bool_spatial_variations = pack_inputs(inputs)

    # Read and simplify dataset
    data_raw = read_BioLogic_data(data_file)
    if inputs["Data"]["simplify"]:
        data = simplify_EIS_dataset(data_raw,
                                    num_frequency=inputs["Data"]["n_frequency"],
                                    frequency_llim=inputs["Data"]["frequency_llim"],
                                    frequency_ulim=inputs["Data"]["frequency_ulim"])
    else:
        data = data_raw

    # Run initial objective function calculation
    objective_weights = inputs["Objective"]["weights"]
    n_particles = inputs["Particle Network"]["n_particles"]
    objective_0, predicted_eis = calculate_objective_function(
        x0, data, n_particles=n_particles, objective_weights=objective_weights,
        bool_spatial_variations=bool_spatial_variations, 
        n_basis=inputs["Function"]["n_basis"], inputs=inputs
    )
    print(f"Initial objective value - {objective_0}")

    # Plot initial prediction
    if inputs["Simulation"]["plot"]:
        plot_types = inputs["Plotting"]["type"].split(",")
        plot_save_dir = os.path.join(save_dir, inputs["Plotting"]["directory"])
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        create_impedance_plot(data, predicted_eis, plot_types, plot_save_dir,
                              plot_name='initial_prediction',
                              extensions=['jpg', 'svg'])

    # Perform parameter optimization if required
    if inputs["Simulation"]["fit"]:
        optimization_args = (data, n_particles, objective_weights, 
                             bool_spatial_variations, 
                             inputs["Function"]["n_basis"], inputs)
        metropolis_hastings_mcmc(x0, delx, calculate_objective_function, 
                                 optimization_args, save_dir,
                                 inputs["Optimization"], inputs["Plotting"])
    
    return
        
