import numpy as np
from scipy.sparse import issparse, csr_matrix

def calculate_equivalent_impedance(admittance_matrix, Vs=1, Vt=0):
    """
    Calculates the equivalent impedance from an admittance matrix, ensuring the
    matrix represents a well-posed physical network.

    The function first validates that the admittance matrix is square, has zeros
    along its diagonal (no self-loop admittance), and is symmetric (the network
    is undirected). It then calculates the equivalent impedance between the
    source node (first node) and the drain node (last node). If the source or
    drain node has no admittance connections to other nodes, it returns an
    equivalent impedance of infinity.

    The function supports input as either a numpy ndarray or a sparse CSR
    matrix.

    Parameters
    ----------
    admittance_matrix : np.ndarray or scipy.sparse.csr_matrix
        The admittance matrix of the network. Must be square, have zeros on the
        diagonal, and be symmetric.
    Vs : complex, optional
        Voltage at the source node (first node). Default is 1.
    Vt : complex, optional
        Voltage at the drain node (last node). Default is 0.

    Returns
    -------
    complex
        The equivalent impedance between the source and drain nodes, or
        infinity if the source or drain node is disconnected.

    Raises
    ------
    ValueError
        If the admittance matrix is not square, does not have zeros on the
        diagonal, or is not symmetric.
    """

    # Convert sparse matrix to dense if necessary
    if issparse(admittance_matrix):
        admittance_matrix = admittance_matrix.toarray() # Validate the admittance matrix

    # Validate checks on input admittance matrix
    if admittance_matrix.shape[0] != admittance_matrix.shape[1]:
        raise ValueError("The admittance matrix must be square.")
    if not np.all(np.isclose(np.diag(admittance_matrix), 0)):
        raise ValueError("The admittance matrix must have zeros on its "
                         "diagonal.")
    if not np.allclose(admittance_matrix, admittance_matrix.T):
        raise ValueError("The admittance matrix must be symmetric.")
    
    # Check if source or drain node has no connections to other nodes
    if (np.all(admittance_matrix[0, :] == 0) 
        or np.all(admittance_matrix[-1, :] == 0)):
        return np.inf

    # Modify Vs for phase adjustment if necessary
    Vs = Vs * np.exp(1j*0.4)  # Uncomment if phase adjustment is required
    num_nodes = admittance_matrix.shape[0]
    mask = np.ones(num_nodes, dtype=bool)
    mask[[0, -1]] = False

    # Create Laplacian matrix by setting diagonal elements to sum of rows
    laplacian_matrix = np.copy(admittance_matrix)
    np.fill_diagonal(laplacian_matrix, -admittance_matrix.sum(axis=1))
    laplacian_matrix = -1 * laplacian_matrix

    # Modified Laplacian matrix excluding source and drain rows/columns
    L_mod = laplacian_matrix[mask, :][:, mask]

    # Calculate source and drain currents considering Vs and Vt
    I = np.zeros(num_nodes-2, dtype=np.cdouble)
    I += admittance_matrix[mask, +0] * Vs
    I += admittance_matrix[mask, -1] * Vt
    
    # Calculate the condition number of A
    cond_number = np.linalg.cond(L_mod)

    # Check if the condition number is too high, indicating singularity
    if cond_number > 1 / 1e-12:
        return np.inf
    
    # Solve for node voltages excluding source and drain
    V_mid = np.linalg.solve(L_mod, I)
    
    # Calculate total current flowing from source to drain
    I_total = np.dot(admittance_matrix[0, 1:], Vs-np.append(V_mid, Vt))
    
    # Calculate equivalent impedance
    Z_eq = (Vs - Vt) / I_total

    return Z_eq