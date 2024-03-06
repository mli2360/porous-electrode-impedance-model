import os
import matplotlib.pyplot as plt

from src.data_io import (read_BioLogic_data,
                         simplify_EIS_dataset,
                         read_config,
                         validate_configuration,
                         pack_inputs)
from src.objective import calculate_objective_function
from src.plot import create_impedance_plot
from src.optimization import metropolis_hastings_mcmc


def main(config_file, data_file):

    # Read in configuration file
    inputs = read_config(config_file)
    validate_configuration(inputs)
    x0, delx, bool_spatial_variations = pack_inputs(inputs)
    
    # Read in dataset
    data_raw = read_BioLogic_data(data_file)
    if inputs["Data"]["simplify"]:
        data = simplify_EIS_dataset(data_raw,
                                    num_frequency=inputs["Data"]["n_frequency"],
                                    frequency_llim=inputs["Data"]["frequency_llim"],
                                    frequency_ulim=inputs["Data"]["frequency_ulim"])
    else:
        data = data_raw

    # Run one iteration of the objective function calculation
    if inputs["Simulation"]["fit"]:
        chain_length = inputs["Optimization"]["chain_length"]
        objective_weights = inputs["Objective"]["weights"]
    else:
        objective_weights = [0.5, 0.5/3, 0.5/3, 0.5/3]
    n_particles = inputs["Particle Network"]["n_particles"]

    objective_0, predicted_eis = calculate_objective_function(
        x0, data, n_particles=inputs["Particle Network"]["n_particles"],
        objective_weights=objective_weights,
        bool_spatial_variations=bool_spatial_variations,
        n_basis=inputs["Function"]["n_basis"],
        inputs=inputs
    )

    # Plot initial system
    if inputs["Simulation"]["plot"]:
        plot_types = inputs["Plotting"]["type"].split(",")
        plot_save_dir = os.path.join(inputs["Simulation"]["save_directory"],
                                     inputs["Plotting"]["directory"])
        # Ensure the directory exists
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)

        create_impedance_plot(data, predicted_eis, plot_types, plot_save_dir,
                              plot_name='initial_prediction',
                              extensions=['jpg', 'svg'])

    # Perform optimization of the parameters
    if inputs["Simulation"]["fit"]:
        args = (data, n_particles, objective_weights, bool_spatial_variations,
                inputs["Function"]["n_basis"], inputs)
        metropolis_hastings_mcmc(x0, delx, calculate_objective_function,
                                 args, inputs["Simulation"]["save_directory"],
                                 inputs["Optimization"], inputs["Plotting"])
        
    return
        
