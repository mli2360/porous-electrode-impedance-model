import os
import numpy as np
import matplotlib.pyplot as plt

def create_impedance_plot(measured_eis, predicted_eis, plot_types, save_dir,
                          plot_name, extensions=['jpg']):
    """
    Generates impedance plots comparing measured and predicted EIS data for
    specified plot types, saves the plots to the specified directory under a
    given name with specified file extensions.

    Parameters
    ----------
    measured_eis : pandas.DataFrame
        DataFrame containing the measured EIS data with fields 'freq/Hz',
        'Re(Z)/Ohm', and '-Im(Z)/Ohm'.
    predicted_eis : pandas.DataFrame
        DataFrame containing the predicted EIS data with fields 'freq/Hz',
        'Re(Z)/Ohm', and '-Im(Z)/Ohm'.
    plot_types : list of str
        List of strings indicating the types of plots to create (e.g.,
        'Nyquist', 'Bode-magnitude', 'Bode-phase').
    save_dir : str
        Directory path where the plots will be saved.
    plot_name : str
        Base name for the plot files.
    extensions : list of str, optional
        List of file extensions for saving plots (default is ['jpg']).

    Raises
    ------
    ValueError
        If an unknown plot type is provided.

    Examples
    --------
    >>> measured_eis = pandas.DataFrame(data={'freq/Hz': [1, 10, 100],
    'Re(Z)/Ohm': [10, 5, 2], '-Im(Z)/Ohm': [5, 10, 15]})
    >>> predicted_eis = pandas.DataFrame(data={'freq/Hz': [1, 10, 100],
    'Re(Z)/Ohm': [8, 6, 3], '-Im(Z)/Ohm': [6, 8, 12]})
    >>> create_impedance_plot(measured_eis, predicted_eis, ['Nyquist',
    'Bode-magnitude'], '/plots', 'impedance_analysis')
    """

    # Create subplots based on the number of plot types
    fig, axes = plt.subplots(1, len(plot_types),
                             figsize=(6 * len(plot_types), 6))
    # Ensure axes is always iterable
    axes = np.atleast_1d(axes)
    
    # Mapping from plot type to plotting function
    plot_functions = {
        'Nyquist': plot_nyquist,
        'Bode-magnitude': plot_bode_magnitude,
        'Bode-phase': plot_bode_phase,
    }

    # Create each requested plot
    for ax, plot_type in zip(axes, plot_types):
        if plot_type in plot_functions:
            plot_functions[plot_type](ax, measured_eis, predicted_eis)
        else:
            raise ValueError(f"No plotting function for '{plot_type}'")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plots with each specified file extension
    for extension in extensions:
        plot_filename = f"{plot_name}.{extension}"
        plot_path = os.path.join(save_dir, plot_filename)
        fig.savefig(plot_path, format=extension)
    
    # Close the figure to free resources
    plt.close(fig)


def plot_nyquist(ax, measured_eis, predicted_eis):
    """
    Plot the Nyquist plot representation for measured and predicted impedance.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the Nyquist plot.
    measured_eis : pandas.DataFrame
        Data frame containing the measured impedance data, with columns
        'Re(Z)/Ohm' and '-Im(Z)/Ohm'.
    predicted_eis : pandas.DataFrame
        Data frame containing the predicted impedance data, with columns
        'Re(Z)/Ohm' and '-Im(Z)/Ohm'.

    Returns
    -------
    None
    """

    ax.plot(measured_eis['Re(Z)/Ohm'], measured_eis['-Im(Z)/Ohm'], 
         'o-', color='black', markersize=4, label='Measured')  # 'o-' creates lines with circle markers
    ax.plot(predicted_eis['Re(Z)/Ohm'], predicted_eis['-Im(Z)/Ohm'], 
         'o-', color='red', markersize=4, label='Predicted')  # 'o-' creates lines with circle markers
    ax.set_xlabel(r"$Z'$ [Ohms]")
    ax.set_ylabel(r"$-Z''$ [Ohms]")
    ax.grid(True)
    ax.axis('equal')  # Ensure the plot is square shaped for correct aspect ratio


def plot_bode_magnitude(ax, measured_eis, predicted_eis):
    """
    Plot the Bode magnitude plot representation for measured and predicted
    impedance.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the Bode magnitude plot.
    measured_eis : pandas.DataFrame
        Data frame containing the measured impedance data, with columns
        'freq/Hz', 'Re(Z)/Ohm', and '-Im(Z)/Ohm'.
    predicted_eis : pandas.DataFrame
        Data frame containing the predicted impedance data, with columns
        'freq/Hz', 'Re(Z)/Ohm', and '-Im(Z)/Ohm'.

    Returns
    -------
    None
    """

    Z_measured = measured_eis['Re(Z)/Ohm'] - 1j*measured_eis['-Im(Z)/Ohm']
    Z_predicted = predicted_eis['Re(Z)/Ohm'] - 1j*predicted_eis['-Im(Z)/Ohm']
    ax.plot(measured_eis['freq/Hz'], np.abs(Z_measured),
            'o-', color='black', markersize=4, label='Measured')
    ax.plot(predicted_eis['freq/Hz'], np.abs(Z_predicted),
            'o-', color='red', markersize=4, label='Predicted')
    ax.set_xlabel(r"$\omega$ [Hz]")
    ax.set_ylabel(r"$|Z| [Ohms]$")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True)


def plot_bode_phase(ax, measured_eis, predicted_eis):
    """
    Plot the Bode phase plot representation for measured and predicted impedance
    data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the Bode phase plot.
    measured_eis : pandas.DataFrame
        Data frame containing the measured impedance data, with columns
        'freq/Hz', 'Re(Z)/Ohm', and '-Im(Z)/Ohm'.
    predicted_eis : pandas.DataFrame
        Data frame containing the predicted impedance data, with columns
        'freq/Hz', 'Re(Z)/Ohm', and '-Im(Z)/Ohm'.

    Returns
    -------
    None
    """

    Z_measured = measured_eis['Re(Z)/Ohm'] - 1j*measured_eis['-Im(Z)/Ohm']
    Z_predicted = predicted_eis['Re(Z)/Ohm'] - 1j*predicted_eis['-Im(Z)/Ohm']
    ax.plot(measured_eis['freq/Hz'], np.angle(Z_measured, deg=True),
            'o-', color='black', markersize=4, label='Measured')
    ax.plot(predicted_eis['freq/Hz'], np.angle(Z_predicted, deg=True),
            'o-', color='red', markersize=4, label='Predicted')
    ax.set_xlabel(r"$\omega$ [Hz]")
    ax.set_ylabel(r"Phase angle $\theta$ [degrees]")
    ax.set_xscale('log')
    ax.grid(True)


