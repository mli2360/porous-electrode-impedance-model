import numpy as np
import os
import pickle

from src.plot import create_impedance_plot

def metropolis_hastings_mcmc(x0, delx, calculate_objective_function, args, 
                             save_directory, settings, settings_plot):
    """
    Perform MCMC sampling using the Metropolis-Hastings algorithm.

    Parameters
    ----------
    x0 : array_like
        Initial parameter vector.
    del_x0 : array_like
        Characteristic step size for the proposal distribution.
    calculate_objective_function : callable
        Objective function to be minimized, should take a parameter vector `x` and additional arguments `*args`.
    args : tuple
        Additional arguments required by the objective function.
    n_samples : int, optional
        Number of MCMC samples to generate (default is 10000).

    Returns
    -------
    samples : ndarray
        Array of sampled parameter vectors.
    acceptance_rate : float
        The acceptance rate of the MCMC sampler.
    """
    
    # Initialize logistical variables
    data = args[0]
    plot_types = settings_plot["type"].split(",")
    plot_save_dir = os.path.join(save_directory, settings_plot["directory"])
    results_file_path = os.path.join(save_directory, 'results.pkl')

    
    # Initialize current and best states
    x_current = x_best = np.array(x0)
    current_objective = best_objective = \
        calculate_objective_function(x_current, *args)[0]
    burn_samples = [x_current]
    n_accept = 0

    # Display starting conditions
    print(f"Burn-in started - Length {settings['burn_length']}")
    print(f"Current Objective Value - {current_objective}")
    print(f"Current Parameters - {x_current}")
    print('\n')

    burn_temperatures = np.exp(np.linspace(np.log(settings["burn_temperature"]),
                                    np.log(settings["temperature"]),
                                    settings["burn_length"]))
    state = "burn"
    # First do mcmc burn-in wiht a higher temperature profile
    for idx in range(settings["burn_length"]):
        # Propose a new state
        x_proposal = x_current + delx * np.random.randn(*x_current.shape)
        proposal_objective, predicted_eis = \
            calculate_objective_function(x_proposal, *args)
        
        # Calculate acceptance probability
        # Assuming a symmetric proposal distribution here
        temperature = burn_temperatures[idx]
        acceptance_probability = np.min((1, np.exp((current_objective - proposal_objective) / temperature)))

        # Accept or reject the new state
        if np.random.rand() < acceptance_probability:
            x_current = x_proposal
            current_objective = proposal_objective
            n_accept += 1

        # Save best states
        if current_objective < best_objective:
            best_objective = current_objective
            x_best = x_current

        burn_samples.append(x_current)

        if idx % settings["display_frequency"] == 0:
            print(f"Burn-in Sample {idx} of {settings['burn_length']}. Temperature {temperature}")
            print(f"Current Objective Value - {current_objective}")
            print(f"Current Parameters - {x_current}")
            print('\n')
        if idx % settings["plot_frequency"] == 0:
            plot_name = state + f"_{idx}_of_{settings['burn_length']}"
            create_impedance_plot(data, predicted_eis, plot_types, plot_save_dir,
                              plot_name=plot_name, extensions=['jpg', 'svg'])
        if idx % settings["save_frequency"] == 0:
            burn_acceptance_rate = n_accept / (idx + 1)
            with open(results_file_path, 'wb') as file:  # 'wb' stands for 'write binary'
                # Package your variables into a dictionary or a list
                variables = {
                    'x_current': x_current, 
                    'current_objective': current_objective, 
                    'x_best': x_best, 
                    'best_objective' : best_objective,
                    'burn_samples' : np.array(burn_samples),
                    'burn_acceptance_rate' : burn_acceptance_rate,
                    'temperature' : temperature,
                }
                # Use pickle.dump() to write the object to the file
                pickle.dump(variables, file)

    n_accept = 0
    samples = [x_current]
    
    state = "sampling"
    for idx in range(settings["chain_length"]):
        # Propose a new state
        x_proposal = x_current + delx * np.random.randn(*x_current.shape)
        proposal_objective = calculate_objective_function(x_proposal, *args)

        # Calculate acceptance probability
        # Assuming a symmetric proposal distribution here
        temperature = settings["temperature"]
        acceptance_probability = np.min((1, np.exp((current_objective - proposal_objective) / temperature)))
        
        # Accept or reject the new state
        if np.random.rand() < acceptance_probability:
            x_current = x_proposal
            current_objective = proposal_objective
            n_accept += 1
        
        # Save best states
        if current_objective < best_objective:
            best_objective = current_objective
            x_best = x_current

        samples.append(x_current)

        if idx % settings["display_frequency"] == 0:
            print(f"Sample {idx} of {settings['chain_length']}. Temperature {temperature}")
            print(f"Current Objective Value - {current_objective}")
            print(f"Current Parameters - {x_current}")
            print('\n')
        if idx % settings["plot_frequency"] == 0:
            plot_name = state + f"_{idx}_of_{settings['chain_length']}"
            create_impedance_plot(data, predicted_eis, plot_types, plot_save_dir,
                              plot_name=plot_name, extensions=['jpg', 'svg'])
        if idx % settings["save_frequency"] == 0:
            acceptance_rate = n_accept / (idx + 1)
            with open(results_file_path, 'wb') as file:  # 'wb' stands for 'write binary'
                # Package your variables into a dictionary or a list
                variables = {
                    'x_current': x_current, 
                    'current_objective': current_objective, 
                    'x_best': x_best, 
                    'best_objective' : best_objective,
                    'burn_samples' : np.array(burn_samples),
                    'burn_acceptance_rate' : burn_acceptance_rate,
                    'samples' : np.array(samples),
                    'acceptance_rate' : acceptance_rate,
                    'temperature' : temperature,
                }
                # Use pickle.dump() to write the object to the file
                pickle.dump(variables, file)

    samples = np.array(samples)
    acceptance_rate = n_accept / settings["chain_length"]

    return samples, acceptance_rate