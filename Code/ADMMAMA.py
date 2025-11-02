# The objective of this module is to implement the ADMM and AMA
# algorithms for solving the convex clustering problem as described in
# "Splitting Methods for Convex Clustering" by Eric C. Chi and Kenneth Lange.

#At this point we have already a module that implements the boosting algorithm
#Proposed on "A new perspective on Boosting in Linear Regression"


#We star implementing the proximal map for several norms
import numpy as np 
from pyproximal import L1, L2, L0


#We start defining the proximal operators for the norms we 
# will use in the ADMM and AMA algorithms
def proximal(x,sigma, tau, norm_type):
    """
    Compute the proximal operator for a given norm.
    
    Parameters:
        x: Input vector.
        coef: Coefficient for the proximal operator.
        tau: Step size.
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

