import numpy as np
from numpy.polynomial.legendre import legval


def evaluate_parameters(x, y0,
                        basis_coefficients=None, function_type='legendre'):
    """
    Evaluates a set of parameters based on the specified function representation
    type and basis function coefficients across spatial positions, adjusted by
    the baseline parameter value.

    Parameters
    ----------
    x : numpy.ndarray
        The 1D numpy array representing the spatial positions at which the
        function is to be evaluated.
    y0 : float
        The baseline parameter value to adjust the function by.
    basis_coefficients : numpy.ndarray, optional
        The coefficients of the basis functions used to represent the spatial
        variation of the parameter. If None, the parameter is considered not to
        vary spatially and y0 is used across all positions.
    function_type : str, optional
        The type of function representation used. Currently only 'legendre' is
        implemented.

    Returns
    -------
    y : numpy.ndarray
        The evaluated parameter values at each position specified in x, adjusted
        by y0.

    Raises
    ------
    ValueError
        If an unsupported function_type is specified.

    Examples
    --------
    >>> x = np.linspace(-1, 1, 100)
    >>> y0 = 5
    >>> coeffs = [1, 2, 3]  # Example coefficients for a Legendre polynomial
    >>> y = evaluate_parameters(x, y0, basis_coefficients=coeffs)
    >>> print(y)
    """

    if basis_coefficients is None:
        # If there are no coefficients, the parameter does not vary spatially.
        y = np.full_like(x, y0)
    else:
        # Validate function type
        if function_type != 'legendre':
            raise ValueError("Only 'legendre' polynomials have been implemented"
                             " for function_type.")

        # Use Legendre polynomial evaluation if coefficients are provided
        function_evaluator = legval
        fun_values = function_evaluator(x, basis_coefficients)

        # Adjust the function values based on the baseline value y0
        y = fun_values + y0 - fun_values[-1]  # Ensure the adjustment is made relative to the end value

    return y
