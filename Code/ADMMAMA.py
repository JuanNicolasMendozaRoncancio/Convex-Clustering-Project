# The objective of this module is to implement the ADMM and AMA
# algorithms for solving the convex clustering problem as described in
# "Splitting Methods for Convex Clustering" by Eric C. Chi and Kenneth Lange.

#At this point we have already a module that implements the boosting algorithm
#Proposed on "A new perspective on Boosting in Linear Regression"


#We star implementing the proximal map for several norms
import numpy as np 
from pyproximal import L1, L2, L0
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


#We start defining the proximal operators for the norms we 
# will use in the ADMM and AMA algorithms
def proximal(x,sigma, tau, norm_type):
    """
    Compute the proximal operator for a given norm.
    
    Parameters:
        x: Input vector.
        coef: Coefficient for the norm.
        tau: Coefficient for the proximal operator.
        norm_type: Type of norm ('L1', 'L2', 'L0').
    """
    if norm_type == 'L1':
        l1 = L1(sigma=sigma)
        return l1.prox(x, tau)
    elif norm_type == 'L2':
        l2 = L2(sigma=sigma)
        return l2.prox(x, tau)
    elif norm_type == 'L0':
        l0 = L0(sigma=sigma)
        return l0.prox(x, tau)
    else:
        raise ValueError("Invalid norm type. Choose 'L1', 'L2', or 'L0'.")   
#The ADMM and AMA algorithms will be dependent 
#on the weight matrix W that defines the graph structure
def built_edges(W):
    """
    Build edges and weights from the weight matrix W.
    Parameters:
        W: Weight matrix.
    Returns:  
        edges: List of edges.
        weights: Corresponding weights.
    """
    n = W.shape[0] # number of nodes
    edges = []
    weights = []
    for i in range(n):
        for j in range(i+1, n): # W is symmetric
            if W[i,j] > 0:
                edges.append((i,j))
                weights.append(W[i,j])

    return edges, np.array(weights, dtype=np.float64)

def build_graph_laplacian(n, edges, weights):
    """
    Construye el Laplaciano L del grafo:
        L = sum_l w_l * (e_i - e_j)(e_i - e_j)^T
    Devuelve L en formato disperso (sparse).
    """
    row, col, data = [], [], []
    for (i, j), w in zip(edges, weights):
        # diagonal terms
        row.extend([i, i, j, j])
        col.extend([i, j, i, j])
        data.extend([w, -w, -w, w])
    L = csr_matrix((data, (row, col)), shape=(n, n))
    return L

#The algorithms in the paper depend on the
# connectednes of the graph, if its fully connected
# we apply a direct formula. If not its another (and more general) case

def ADMMfullyconected(X,gamma,W = None, nu = 1,
                      max_iter = 1000, tol = 1e-5,
                      norm_type = 'L2', verbose = False):
    """
    ADMM algorithm for convex clustering with fully connected graph.
    Parameters:
        X: Data matrix.
        gamma: Regularization parameter.
        W: Weight matrix.
        nu: Augmented Lagrangian parameter.
        max_iter: Maximum number of iterations.
        tol: Tolerance for convergence.
        norm_type: Type of norm for proximal operator.
        verbose: If True, print progress.
    Returns:
        U: Clustered data matrix.
        V: Auxiliary variable matrix.
        lambda: Lagrange multipliers.
        history: Convergence history.
    """
    p, n = X.shape
    # Edges list and weights:
    edges = [(i,j) for i in range(n) for j in range(i+1,n)]
    m = len(edges)
    if W is None:
        weights = np.ones(m)
    else:
        weights = np.array(W, dtype=np.float64)

    # Initialize variables
    V = np.zeros((p, m))
    lambda_ = np.zeros((p, m))

    history = {'primal_residual': [], 'obj': []}

    # Precompute mean of data
    xbar = np.mean(X, axis=1, keepdims=True)

    for it in range(1, max_iter+1):
        Y = X.copy()
        # Compute yi
        for idx, (i,j) in enumerate(edges):
            vt = lambda_[:,idx] + nu*V[:,idx]
            Y[:,i] -= vt
            Y[:,j] += vt
        
        # Update U
        U = (1/(1 + n*nu))*Y + (n*nu/(1 + n*nu))*np.repeat(xbar, n, axis=1)

        # Update V
        for idx, (i,j) in enumerate(edges):
            diff = U[:,i] - U[:,j] - (1/nu)*lambda_[:,idx]
            V[:,idx] = proximal(diff, 1, gamma*weights[idx]/nu, norm_type)
            lambda_[:,idx] += nu*(V[:,idx] - U[:,i] + U[:,j])

        primal_sq = 0
        for idx, (i,j) in enumerate(edges):
            res = V[:,idx]-U[:,i] + U[:,j]  
            primal_sq += np.sum(res**2)
        primal_residual = np.sqrt(primal_sq)
        history['primal_residual'].append(primal_residual)


                # optional objective (expensive for large n)
        fit = 0.5 * np.sum((X - U)**2)
        pen = 0.0
        for idx, (i, j) in enumerate(edges):
            pen += weights[idx] * np.linalg.norm(U[:, i] - U[:, j])
        history['obj'].append(fit + gamma * pen)

        if verbose and (it == 1 or it % 50 == 0):
            print(f"it {it:4d} | primal_res {primal_residual:.3e} | obj {history['obj'][-1]:.6f}")

        if primal_residual < tol:
            if verbose:
                print(f"Converged at iter {it}: primal_res={primal_residual:.2e}")
            break

        return U, V, lambda_, history

def ADMMgeneral(X, edges, weights, gamma, nu=1,
                max_iter=1000, tol=1e-5,
                norm_type='L2', verbose=False):
    """
    ADMM algorithm for convex clustering with general graph structure.
    Parameters:
        X: Data matrix.
        edges: List of edges.
        Weigths: Corresponding weights.
        gamma: Regularization parameter.
        nu: Augmented Lagrangian parameter.
        max_iter: Maximum number of iterations.
        tol: Tolerance for convergence.
        norm_type: Type of norm for proximal operator.
        verbose: If True, print progress.
    Returns:
        U: Clustered data matrix.
        V: Diference variable matrix.
        lambda: Lagrange multipliers.
    """
    p, n = X.shape
    m = len(edges)
    Weigths = np.array(Weigths, dtype=np.float64)

    # Initialize variables
    V = np.zeros((p, m))
    lambda_ = np.zeros((p, m))
    U = X.copy()

    # build Laplacian and matrix M
    L = build_graph_laplacian(n, edges, weights)
    M = (csr_matrix(np.eye(n)) + nu * L)  # matriz (I + νL)

    for it in range(1, max_iter + 1):
        # Step 1: actualitation of Y
        Y = X.copy()
        for idx, (i, j) in enumerate(edges):
            vt = lambda_[:, idx] + nu * V[:, idx]
            Y[:, i] += vt
            Y[:, j] -= vt

        # Step 2: actualitation of U
        U = np.vstack([spsolve(M, Y[k, :]) for k in range(p)])

        # Step 3: actualitation of V and lambda
        for idx, (i, j) in enumerate(edges):
            z = U[:, i] - U[:, j] - (1.0 / nu) * lambda_[:, idx]
            sigma_l = (gamma * weights[idx]) / nu
            V[:, idx] = proximal(z, 1, sigma_l, norm_type)
            lambda_[:, idx] += nu * (V[:, idx] - U[:, i] + U[:, j])

        primal_res = np.sqrt(sum(np.linalg.norm(V[:, idx] - U[:, i] + U[:, j])**2
                                 for idx, (i, j) in enumerate(edges)))

        if verbose and (it == 1 or it % 50 == 0):
            print(f"Iter {it:4d} | Primal residual: {primal_res:.3e}")

        if primal_res < tol:
            if verbose:
                print(f"Converged at iteration {it} (residual = {primal_res:.2e})")
            break

    return U, V, lambda_

